from elasticsearch import Elasticsearch
from elasticsearch import helpers
import os
import cv2
import numpy as np
import torch.nn as nn
import torch
import torchvision
import math
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import json
import pytesseract
import time
es = Elasticsearch()

#---------------------------------------------------------------------------------------------------------------#
'''
**important functions of Main()**
1. Main(image_path):
    code structure see ==> "Abbildungen\\single_image_processing.svg"

2. TiltCorrection(image):
    - input: the image we want to tilt correct, must be gray image
    - output: the tilt corrected image

    at first call the function GetAngle(),witch basic of cv2.minAreaRect(), to get the tilt angle
    then call ImageRotate() to correct the tilt of image

3. SizeNormalize(image):
    Normalize the input image size to 1024 x 1024
    - input: image, 1 channel or 3 channels
    - output: image 1024 x 1024m 3 channels

4. PositionTable(img_1024, img_path):
    get the position of table bigger than 80000 pixels in a image
    - input 1: image must be 3 channel, 1024 x 1024
    - input 2: the path of the image
    - input 3: the used model ---> 'densenet' or 'unet'
    - output: the location of tables in image [[x, y, w, h], ..] here x and y are the locaiton of top left point

5. GetTableZone(table_boundRect, img_1024):
    ROI the table in image
    - input 1: location of tables in image
    - input 2: image
    - output: table_zone

6. DeletLines(img)
    delet all the lines on image
    - input: gray image
    - output: bina image without lines

    at first call the function LSDGetLines() to mark all lines on image
    then in function OrImage() mit cv2.bitwise_or to delet all the lines

7. GetCell(img_deletline):
    Get word blocks by dilate
    - input: bina image without lines
    - output: contour location of text blocks

8. GetLabel(location):
    Assign row and column labels to each cell
    - input: the location of cells (not aligned), [[x,y,w,h], ..] here x and y are the locaiton of top left point of cell
    - output1: the center locationn of cells (aligned), [[center_x, center_y, w, h, x, y], ..]
    - output2: the label of each cell, [[row?, col?], ..]
    - output3: the size of table, [[row_number, col_number], [...], ...]

    during this function the function PointCorrection will be called to align the points

9. ReadCell(center_list, image):
    ORI of each cell and OCR by tesseract
    - input 1: center_list of the table
    - input 2: the image with table
    - output: infos in each cell

10. GetDataframe(list_info, label_list, tablesize):
    Rebuild the table in a Dataframe
    - input 1: list_info
    - input 2: label_list
    - input 3: tablesize
    - output: Dataframe

11. WriteData(df, label_):
    write dataframe to elasticsearch
    - input 1: dataframe
    - input 2: label_, here is the table name, for example: table_2_of_table2_rotate_0

12.Umform(df_dict, label_):
    see issue: instraction to funciton Umform()
    - input 1: df_dict is a dict, in it is detected table
    - input 2: label_ is the quelle of the table
    - output: processed table information, key is header, value is value in columen form

13. Search(index_, label_):
    Searches for data in ES-index, for example: table_2_of_table2_rotate_0
    - input 1: index_ is 'table'
    - input 2: label of table, for example: table_2_of_table2_rotate_0, if label_ is all --> back all datas
    - output: result

'''
#---------------------------------------------------------------------------------------------------------------#
# model structures

# densenet- tablenet


class DenseNet(nn.Module):
    def __init__(self, pretrained=True, requires_grad=True):
        super(DenseNet, self).__init__()
        denseNet = torchvision.models.densenet121(pretrained=True).features
        self.densenet_out_1 = torch.nn.Sequential()
        self.densenet_out_2 = torch.nn.Sequential()
        self.densenet_out_3 = torch.nn.Sequential()

        for x in range(8):
            self.densenet_out_1.add_module(str(x), denseNet[x])
        for x in range(8, 10):
            self.densenet_out_2.add_module(str(x), denseNet[x])

        self.densenet_out_3.add_module(str(10), denseNet[10])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):

        out_1 = self.densenet_out_1(x)  # torch.Size([1, 256, 64, 64])
        out_2 = self.densenet_out_2(out_1)  # torch.Size([1, 512, 32, 32])
        out_3 = self.densenet_out_3(out_2)  # torch.Size([1, 1024, 32, 32])
        return out_1, out_2, out_3


class TableDecoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(TableDecoder, self).__init__()
        self.conv_7_table = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=kernels[0],
            stride=strides[0])
        self.upsample_1_table = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=kernels[1],
            stride=strides[1])
        self.upsample_2_table = nn.ConvTranspose2d(
            in_channels=128 + channels[0],
            out_channels=256,
            kernel_size=kernels[2],
            stride=strides[2])
        self.upsample_3_table = nn.ConvTranspose2d(
            in_channels=256 + channels[1],
            out_channels=1,
            kernel_size=kernels[3],
            stride=strides[3])

    def forward(self, x, pool_3_out, pool_4_out):
        x = self.conv_7_table(x)  # [1, 256, 32, 32]
        out = self.upsample_1_table(x)  # [1, 128, 64, 64]
        out = torch.cat((out, pool_4_out), dim=1)  # [1, 640, 64, 64]
        out = self.upsample_2_table(out)  # [1, 256, 128, 128]
        out = torch.cat((out, pool_3_out), dim=1)  # [1, 512, 128, 128]
        out = self.upsample_3_table(out)  # [1, 1, 1024, 1024]
        return out


class TableNet(nn.Module):
    def __init__(self, encoder='densenet', use_pretrained_model=True, basemodel_requires_grad=True):
        super(TableNet, self).__init__()

        self.base_model = DenseNet(
            pretrained=use_pretrained_model, requires_grad=basemodel_requires_grad)
        self.pool_channels = [512, 256]
        self.in_channels = 1024
        self.kernels = [(1, 1), (1, 1), (2, 2), (16, 16)]
        self.strides = [(1, 1), (1, 1), (2, 2), (16, 16)]

        # common layer
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=256, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8))

        self.table_decoder = TableDecoder(
            self.pool_channels, self.kernels, self.strides)

    def forward(self, x):

        pool_3_out, pool_4_out, pool_5_out = self.base_model(x)
        conv_out = self.conv6(pool_5_out)  # [1, 256, 32, 32]
        # torch.Size([1, 1, 1024, 1024])
        table_out = self.table_decoder(conv_out, pool_3_out, pool_4_out)
        return table_out

#---------------------------------------------------------------------------------------------------------------#
# functions


def FLDGetLines(img, minLong):
    '''
    lines be marked by FLD

    - input 1: is a bina image
    - input 2: the min long of lines

    - output 1: is a new black image with same shape of input image, on it is the lines of image, location to location
                be used for DeleteLine
    - output 2: list of lines, [[x0,y0,x1,y1],[x2,y2,x3,y3],[...],[...],...], for tiltCorrection

    '''

    # make a new black image with the same shape of input img
    copy_image = np.zeros((img.shape[0], img.shape[1]))

    fld = cv2.ximgproc.createFastLineDetector()
    # get all the location of lines by FLD, if no line, dlines = None
    dlines = fld.detect(img)
    # print(dlines)
    longLines = []
    if dlines is not None:
        for dline in dlines:
            x0 = int(round(dline[0][0]))
            y0 = int(round(dline[0][1]))
            x1 = int(round(dline[0][2]))
            y1 = int(round(dline[0][3]))
            long = (y1-y0)*(y1-y0)+(x1-x0)*(x1-x0)
            if long >= minLong*minLong:
                # It is possible that the shorter lines are part of the letter, so filter out the longer lines.
                cv2.line(copy_image, (x0, y0), (x1, y1), color=255,
                         thickness=3, lineType=cv2.LINE_AA)  # draw the white line on black image
                longLines.append([x0, y0, x1, y1])

    if __name__ == "__main__":
        plt.subplot(121)
        plt.imshow(img, cmap='gray')
        plt.subplot(122)
        plt.imshow(copy_image)
        plt.show()

    return copy_image, longLines


def GetLineAngle(img):
    '''
    - input: gray image
    - output: angle for tilt correction

    '''
    bina_image = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 10)  # Gaussian binar
    minLong = (img.shape[0]+img.shape[1])//4
    longLines = FLDGetLines(bina_image, minLong)[1]

    if len(longLines) == 0:
        return 'nolines'
    else:
        angle_list = []
        for line in longLines:
            x0, y0, x1, y1 = line
            if x0 == x1:
                angle_list.append(90)
            elif y0 == y1:
                angle_list.append(0)
            else:
                t = float(y1-y0)/(x1-x0)
                rotate_angle = (math.degrees(math.atan(t)))
                angle_list.append(rotate_angle)

        angle_list_45 = [angle_list[i] for i in range(len(angle_list)) if abs(
            angle_list[i]) < 45]  # list for angle < +-45
        # print(angle_list_45)
        # print(angle_list)

        if len(angle_list_45) == 0:  # if no abs(anlge) < 45
            angle_average = 0
        else:
            angle_average = (sum(angle_list_45)/len(angle_list_45))

        return angle_average


def GetBoxAngle(img):
    '''
    - input: img, is gray

    - output: angle need to tilt correction

    '''
    bina_image = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 10)  # Gaussian binar
    bina_image1 = cv2.bitwise_not(bina_image)  # invert the image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    bina_image1 = cv2.erode(bina_image1, kernel, iterations=1)
    # noise reduce by opening
    bina_image1 = cv2.dilate(bina_image1, kernel, iterations=1)

    # get the location of white pixel
    coords = np.where(bina_image1 > 0)

    points = [None]*len(coords[0])
    for i, x in enumerate(coords[0]):
        y = coords[1][i]
        points[i] = (y, x)
    points = np.array(points)
    rect = cv2.minAreaRect(points)
    angle = (rect[2])  # round all the white pixel by a rect

    # https://theailearner.com/tag/cv2-minarearect/
    # this function has three return
    # [0] -- center point of recttangle
    # [1] -- (w,h) of rectangle
    # [2] -- The rotation angle of the rectangle,
    # angle from the x-axis counterclockwise to w, the range is [-90,0)
    # actually after opencv4.5 is the angle in (0, 90]

    if __name__ == '__main__':
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.polylines(bina_image, [box], True, 0, 3)
        plt.subplot(121)
        plt.imshow(bina_image1, cmap='gray')
        plt.subplot(122)
        plt.imshow(bina_image, cmap='gray')
        plt.show()

    if angle > 45:
        angle = angle - 90

    return angle


def ImageRotate(image, angle):
    '''
    - input 1: the image we want to rotate

    - input 2: rotate anglel, angle is positive for anti-clockwise and negative for clockwise.

    - output: the rotated image

    '''

    # get the shape of the image and then determine the rotate center

    h, w = image.shape[0:2]
    center_X, center_Y = w // 2, h // 2

    # make the rotation matrix
    M = cv2.getRotationMatrix2D((center_X, center_Y), angle, 1.0)
    # https://www.geeksforgeeks.org/python-opencv-getrotationmatrix2d-function/

    # adaptive image border size
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (new_w / 2) - center_X
    M[1, 2] += (new_h / 2) - center_Y

    # perform the actual rotation and return the image
    # borderValue
    image_rotate = cv2.warpAffine(
        image, M, (new_w, new_h), borderValue=(255, 255, 255))
    return image_rotate


def WhiteBordersRemove(gray_image):
    '''
    remove excess white edges of image

    - input: must be grat image

    - output: gray image with reasonably sized white border white border

    '''

    # at first add border for debug of removing
    gray_image1 = cv2.copyMakeBorder(
        gray_image, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
    bina_image = cv2.adaptiveThreshold(
        gray_image1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 10)  # Gaussian binar

    bina_image1 = cv2.bitwise_not(bina_image)  # invert the image

    # round all the white pixel by a rect
    x, y, w, h = cv2.boundingRect(bina_image1)
    thickness = 25

    text_zone = np.ones((h+2*thickness, w+2*thickness, 1))

    text_zone = gray_image1[(y-thickness):(y+h+thickness),
                            (x-thickness):(x+w+thickness)]

    return text_zone


def TiltCorrection(img):
    '''
    - input: the image we want to tilt correct, must be gray image
    - output: the tilt corrected image

    '''
    angle = GetLineAngle(img)
    if angle == 'nolines':  # if no lines in image, then use GetBoxAngle
        angle = GetBoxAngle(img)

    image_rotate = ImageRotate(img, angle)

    if abs(angle) > 25:
        image_rotate = WhiteBordersRemove(image_rotate)

    return image_rotate


def SizeNormalize(img):
    '''
    Normalize the input image size to 1024 x 1024

    - input: image, 1 channel or 3 channel

    - output: image 1024 x 1024

    '''
    # at first normalize the shape to 1024 X () if size > 1024
    shape_list = list(img.shape)

    if max(shape_list) > 1024:
        scaling_r = 1024/max(shape_list)
        # w, h to avoid exceeding 1024, subtract one
        shape_new = [int(shape_list[1]*scaling_r-1),
                     int(shape_list[0]*scaling_r-1)]
        shape_new[shape_new.index(max(shape_new))] = int(1024)
        image = cv2.resize(img, shape_new)

        if __name__ == '__main__':
            print('image_shape ==> ' + str(img.shape) +
                  ' new ==> ' + str(image.shape))
    else:
        image = img
    # The model can only process pictures of 1024x1024 size,
    # so it is necessary to fill the edges of pictures smaller than this size

    h = image.shape[0]
    w = image.shape[1]
    top = (1024-h)//2
    bottom = 1024-h-top
    left = (1024-w)//2
    right = 1024-w-left
    img_1024 = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    img_1024 = np.array(Image.fromarray(
        cv2.cvtColor(img_1024, cv2.COLOR_BGR2RGB)))

    return img_1024


def PositionTable(img_1024, img_path, model_used):
    '''
    get the position of table in a image

    - input 1: image must be 3 channel, 1024 x 1024
    - input 2: the path of the image
    - input 3: the used model

    - output: the location of tables in image [[x, y, w, h], ..] here x and y are the locaiton of top left point

    '''

    device = 'cpu'

    if model_used == 'densenet':
        path = 'Development\\models\\densetable_210.pkl'
        model = torch.load(path, map_location=torch.device(device))

    elif model_used == 'unet':
        path = "Development\\models\\unet100_180spe.pkl"

        model = torch.load(path, map_location=torch.device(device))

    transform = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255,
        ),
        ToTensorV2()
    ])

    image = transform(image=img_1024)["image"]
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)

        if model_used == 'unet':
            pred = model(image)
            pred = (pred.cpu().detach().numpy().squeeze())

        elif model_used == 'densenet':
            pred = model(image)
            pred = torch.sigmoid(pred)
            pred = (pred.cpu().detach().numpy().squeeze())

    pred[:][pred[:] > 0.5] = 255.0
    pred[:][pred[:] < 0.5] = 0.0
    pred = pred.astype('uint8')

    # get contours of the prognose to get tables
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    pred = cv2.erode(pred, kernel, iterations=1)
    pred = cv2.dilate(pred, kernel, iterations=4)  # remove small zone
    pred = cv2.erode(pred, kernel, iterations=3)

    contours, _ = cv2.findContours(
        pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    table_contours = []
    # remove bad contours
    for c in contours:
        if cv2.contourArea(c) > 3000:  # the size of table must be bigger than 3000 pixels
            table_contours.append(c)

    if __name__ == '__main__':
        if len(table_contours) == 0:
            print("No Table Detected ==> " + str(img_path))
        else:
            print('Current Image ==> ' + str(img_path))

    table_boundRect = [None]*len(table_contours)
    for i, c in enumerate(table_contours):
        polyline = cv2.approxPolyDP(c, 5, True)
        table_boundRect[i] = cv2.boundingRect(polyline)

    table_boundRect = sorted(table_boundRect, key=lambda x: x[1])
    # Überlappende Tabellen werden zu einer zusammengeführt:
    mask_image = np.zeros((1024, 1024, 1), np.uint8)
    for x, y, w, h in table_boundRect:
        size = 0
        triangle = np.array(
            [[x-size, y-size], [x-size, y+h+size], [x+w+size, y+h+size], [x+w+size, y-size]])
        cv2.fillConvexPoly(mask_image, triangle, 255)
    contours, _ = cv2.findContours(
        mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    table_contours = []
    # remove bad contours
    for c in contours:
        # the size of table must be bigger than 80000 pixels
        if cv2.contourArea(c) > 40000:
            table_contours.append(c)

    table_boundRect = [None]*len(table_contours)
    for i, c in enumerate(table_contours):
        polyline = cv2.approxPolyDP(c, 5, True)
        table_boundRect[i] = cv2.boundingRect(polyline)

    table_boundRect = sorted(table_boundRect, key=lambda x: x[1])

    # draw bounding boxes
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
    i = 0
    color_image = np.ones((1024, 1024, 3), np.uint8)*255
    for x, y, w, h in table_boundRect:
        size = 0
        # cv2.rectangle(img, (x,y),(x+w,y+h), color, thickness)
        triangle = np.array(
            [[x-size, y-size], [x-size, y+h+size], [x+w+size, y+h+size], [x+w+size, y-size]])
        cv2.fillConvexPoly(color_image, triangle, color[i])
        i += 1
        if i > 3:
            i = 0

    image_add = cv2.addWeighted(img_1024, 0.9, color_image, 0.5, 0)

    if __name__ == '__main__':
        plt.subplot(221)
        plt.title('Input Image 1024x1024')
        plt.imshow(img_1024)
        plt.subplot(222)
        plt.title('Output Prognose')
        plt.imshow(pred, cmap='gray')
        plt.subplot(223)
        plt.title('Verarbeitete Prognose')
        plt.imshow(mask_image, cmap='gray')
        plt.subplot(224)
        plt.title('Tablebreich')
        plt.imshow(image_add)
        plt.show()
        plt.close()

    return table_boundRect


def GetTableZone(table_boundRect, img_1024):
    '''
    ROI the table in image

    - input 1: location of tables in image
    - input 2: image

    - output: table_zone

    '''

    table_zone = [None]*len(table_boundRect)
    for ii, (x, y, w, h) in enumerate(table_boundRect):
        t = 50

        table_zone[ii] = np.ones((h, w, 3))

        table_zone[ii] = cv2.copyMakeBorder(img_1024[(y):(
            y+h), (x):(x+w)], t, t, t, t, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    return table_zone


def OrImage(img1, img2):
    '''
    add two images, weiss + weiss = weiss, weiss + schwarz = weiss, schwarz + schwarz = schwarz

    '''
    img1 = np.array(img1, np.uint8)
    img2 = np.array(img2, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img2 = cv2.dilate(img2, kernel, iterations=1)
    img2 = cv2.erode(img2, kernel, iterations=1)  # closing
    image = cv2.bitwise_or(img1, img2)
    # bitwise_or the original img and the copy img with black lines
    # so can the lines on image be deleted

    return image


def DeletLines(img):
    '''
    delet all the lines on image

    - input: gray image

    - output: bina image without lines

    '''
    img1 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 15)
    # _, img1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    img2 = FLDGetLines(img1, minLong=18)[0]

    img_deletline = OrImage(img1, img2)

    return img_deletline


def GetCell(image_table, img_deletline):
    '''
    Get word blocks by dilate
    - input 1: gray image for OCR
    - input 2: bina image without lines

    - output: contour location of text blocks

    '''

    img_deletline_inv = cv2.bitwise_not(img_deletline)
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    bina_image = cv2.erode(img_deletline_inv, kernel, iterations=1)
    # reduce the noise
    '''

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 3))
    bina_image = cv2.dilate(img_deletline_inv, kernel, iterations=1)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    bina_image = cv2.erode(bina_image, kernel, iterations=1)

    _, bina_image = cv2.threshold(
        bina_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(
        bina_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # round the text zone by rect

    # image_copy = cv2.bitwise_not(img_deletline_inv)

    mask_image = np.zeros(
        (img_deletline_inv.shape[0], img_deletline_inv.shape[1], 1), np.uint8)
    size = 2
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 4 and h > 5 and w < 0.5*img_deletline_inv.shape[1]:
            x = int(x - size)
            y = int(y)
            w = int(w + 2 * size)
            h = int(h)

            # cv2.rectangle(color_image, (x, y), (x+w, y+h), 0, 2)  # round the text zone by rect
            triangle = np.array([[x, y], [x, y+h], [x+w, y+h], [x+w, y]])
            cv2.fillConvexPoly(mask_image, triangle, 255)

    if __name__ == '__main__':
        cv2.imshow("cell closing", bina_image)
        cv2.imshow('mask', mask_image)
        cv2.waitKey()

    contours, _ = cv2.findContours(
        mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # round the text zone by rect
    list_contours = []
    size = 2
    color = (255, 0, 0)
    color_image = np.ones(
        (img_deletline_inv.shape[0], img_deletline_inv.shape[1], 3), np.uint8)*255
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 0.7*img_deletline_inv.shape[1]:
            x = int(x)
            y = int(y - size)
            w = int(w)
            h = int(h + 2 * size)
            list_contours.append((x, y, w, h))
            # cv2.rectangle(color_image, (x, y), (x+w, y+h), 0, 2)  # round the text zone by rect
            triangle = np.array([[x, y], [x, y+h], [x+w, y+h], [x+w, y]])
            cv2.fillConvexPoly(color_image, triangle, color)
    # image_table = cv2.cvtColor(image_table, cv2.COLOR_GRAY2BGR)
    image_add = cv2.addWeighted(image_table, 0.9, color_image, 0.5, 0)

    location = np.array(list_contours)

    return location, image_add  # image_add is used only for show


def GetColumn(table, model_used):
    '''
    use ML modell to get all the the center line of the table columns.
    The red line is the center line of the table column detected by machine
    learning, and the cells that are inside the green lines on either
    side of the red line are grouped into one column.

    '''

    device = 'cpu'

    if model_used == 'densenet':
        path = 'Development\\models\\densecol_140.pkl'
        model = torch.load(path, map_location=torch.device(device))

    elif model_used == 'unet':
        path = "Development\\models\\unetcol_300.pkl"
        model = torch.load(path, map_location=torch.device(device))

    shape_list = list(table.shape)
    scaling_r = 1
    if max(shape_list) > 1024:
        scaling_r = 1024/max(shape_list)
        # w, h to avoid exceeding 1024, subtract one
        shape_new = [int(shape_list[1]*scaling_r-1),
                     int(shape_list[0]*scaling_r-1)]
        shape_new[shape_new.index(max(shape_new))] = int(1024)
        table = cv2.resize(table, shape_new)

    h = table.shape[0]
    w = table.shape[1]
    top = int((1024-h)//2)
    bottom = 1024-h-top
    left = int((1024-w)//2)
    right = 1024-w-left
    img_1024 = cv2.copyMakeBorder(
        table, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    img_1024 = np.array(Image.fromarray(
        cv2.cvtColor(img_1024, cv2.COLOR_BGR2RGB)))

    transform = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255,
        ),
        ToTensorV2()
    ])

    image = transform(image=img_1024)["image"]

    with torch.no_grad():
        image = image.to(device).unsqueeze(0)

        if model_used == 'unet':
            pred = model(image)
            pred = (pred.cpu().detach().numpy().squeeze())

        elif model_used == 'densenet':
            pred = model(image)
            pred = torch.sigmoid(pred)
            pred = (pred.cpu().detach().numpy().squeeze())

    pred[:][pred[:] > 0.5] = 255.0
    pred[:][pred[:] < 0.5] = 0.0
    pred = pred.astype('uint8')

    # get contours of the prognose to get tables
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 19))
    pred = cv2.erode(pred, kernel, iterations=2)
    pred = cv2.dilate(pred, kernel, iterations=1)  # remove small zone

    contours, _ = cv2.findContours(
        pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    col_contours = []
    # remove bad contours
    for c in contours:
        if cv2.contourArea(c) > 500:  # the size of table must be bigger than 500 pixels
            x, y, w, h = cv2.boundingRect(c)
            col_contours.append((int(x+w//2)-left, int(w)))

            cv2.line(img_1024, (int(x+w//2), int(y)), (int(x+w//2), int(y+h)), color=(255, 0, 0),
                     thickness=3, lineType=cv2.LINE_AA)  # draw the white line on black image

    if __name__ == '__main__':
        plt.subplot(131)
        plt.imshow(pred)
        plt.subplot(132)
        plt.imshow(img_1024)

    col_contours = sorted(col_contours, key=lambda x: x[0])
    n = 0
    while n <= 4:
        # print(col_contours)
        for i in range(len(col_contours)-1):
            if abs(col_contours[i+1][0]-col_contours[i][0]) < 0.7*(col_contours[i+1][1]+col_contours[i][1])//2:
                col_contours[i] = ((col_contours[i+1][0]+col_contours[i][0]) //
                                   2, (col_contours[i+1][1]+col_contours[i][1])//2)
                col_contours[i+1] = col_contours[i]
            else:
                continue
        n += 1
        # print(n)
        col_contours = list(set(col_contours))
        col_contours = sorted(col_contours, key=lambda x: x[0])
        # print(col_contours)

    if __name__ == '__main__':
        for col, w in col_contours:
            cv2.line(img_1024, (col+left, top), (col+left, 1024-bottom), color=(0, 0, 255),
                     thickness=3, lineType=cv2.LINE_AA)  # draw the white line on black image
        plt.subplot(133)
        plt.imshow(img_1024)
        plt.show()
    # print(col_contours)
    for i, (col, w) in enumerate(col_contours):
        col_contours[i] = (int(col//scaling_r), int(w//scaling_r))
    # print(col_contours)
    return col_contours


def PointCorrection(location, col_contours):
    '''
    align the cells

    '''

    # location = [dot1, dot2, dot3, dot4, ...]

    #   --------------> x-axis
    # : ##################################
    # : # dot1----dot2-----dot3-----dot4 #
    # ; #  |        |       |         |  #
    # y # dot5----dot6-----dot7-----dot8 #                         x    y    w    h
    #   ##################################                      [[ 16  18]
    #                                                            [ 16 374]
    #                                                            [ 16 640]
    #                                                            [ 16 906]
    #                                                            [ 64 374]
    #                                                            [ 64 640]
    #                                                            [ 64 906]
    #                                                            [ 65  18]  <------- bug !!!

    location = sorted(location, key=lambda x: x[0])
    processed = []

    for i, cell in enumerate(location):    
        # die Zellen, derren Zentrum sich auf beiden Seiten der roten Linie innerhalb 
        # der grünen Linien befinden, werden in einer Spalte gruppiert
        for col, w in col_contours:
            if abs(cell[0] - col) <= w//2:
                cell[0] = col
                processed.append(i)
                break

    for i, cell in enumerate(location):
        # die Zellen, derren Zentrum sich außer grün Linien, werden nach links verschoben.
        if i not in processed:
            if i == 0:
                cell[0] = location[1][0]
            else:
                cell[0] = location[i-1][0]

    location = sorted(location, key=lambda x: (x[1], x[0]))

    for i in range(len(location)-1):
        if location[i+1][1] == location[i][1]:
            continue
        else:
            # suppose there are no two cells with distance less than - in y axis
            if abs(location[i+1][1]-location[i][1]) < int(location[i+1][3]*0.3 + location[i][3]*0.7 - 2):
                location[i+1][1] = location[i][1]
            else:
                continue

    # von oben nach unten zuerst, dann dabei links nach rechts
    location = sorted(location, key=lambda x: (x[1], x[0]))

    return location


def GetLabel(location, col_contours):
    '''
    Assign row and column labels to each cell

    - input: the location of cells (not aligned), [[x,y,w,h], ..] here x and y are the locaiton of top left point of cell

    - output1: cells (aligned), [[center_x, center_y, w, h, x, y], ..]
    - output2: the label of each cell, [[row?, col?], ..]
    - output3: the size of table, [[row_number, col_number], [...], ...]

    '''

    label_list = [None]*len(location)

    # get center of cells
    center_list = [None]*len(location)
    for iii, (x, y, w, h) in enumerate(location):
        center_list[iii] = [x+w//2, y+h//2, w, h, x, y]

    center_list = PointCorrection(center_list, col_contours)
    # print(center_list)
    cols_list = list(set([pp[0] for pp in center_list]))  # alle x axis
    cols_list.sort()
    # print(cols_list)
    rows_list = list(set([pp[1] for pp in center_list]))  # alle y axis
    rows_list.sort()
    # print(rows_list)
    tablesize = [len(rows_list), len(cols_list)]

    for i, (c_x, c_y, w, h, x, y) in enumerate(center_list):
        # label_list[i] = ['row%s' % (rows_list.index(c_y))]

        label_list[i] = [int(rows_list.index(c_y))]
        label_list[i].append('col%s' % (cols_list.index(c_x)))

    return center_list, label_list, tablesize


def Extrakt_Tesseract(image_cell):
    '''
    OCR of a image

    - input: image

    - output: str

    '''

    # pytesseract.pytesseract.tesseract_cmd = 'D:\\for_tesseract\\tesseract.exe'
    result = pytesseract.image_to_string(
        image_cell, lang='deu', config='--psm 7')
    # print(result)

    if '\n' in result:
        result = result.replace('\n', '').replace('|', '').replace('/', '')
    if result == '':
        result = '(unknown)'
    return result


def ReadCell(center_list, image):
    '''
    ORI of each cell and OCR by tesseract

    - input 1: center_list of the table
    - input 2: the image with table for OCR

    - output: infos in each cell

    '''

    size = 0

    list_info = []

    for c_x, c_y, w, h, x, y in center_list:

        cell_zone = np.ones((h+2*size, w+2*size, 1))
        cell_zone = image[(y-size):(y+h+size), (x-size):(x+w+size)]
        cell_zone = cv2.resize(
            cell_zone, (cell_zone.shape[1]*4, cell_zone.shape[0]*4))

        value = [None, None, None]
        for cha in range(3):
            value[cha] = (np.mean(cell_zone[cha], axis=0)[0]
                          + np.mean(cell_zone[cha], axis=0)[-1]
                          + np.mean(cell_zone[cha], axis=1)[0]
                          + np.mean(cell_zone[cha], axis=1)[-1]) // 4

        cell = cv2.copyMakeBorder(
            cell_zone, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=value)

        result = Extrakt_Tesseract(cell)

        # cv2.imshow('',cell)
        # cv2.waitKey()
        # print(result)
        list_info.append(result)

    return list_info


def GetDataframe(list_info, label_list, tablesize):
    '''
    Rebuild the table in a Dataframe

    - input 1: list_info
    - input 2: label_list
    - input 3: tablesize

    - output: Dataframe

    '''
    keys = ['col%s' % s for s in range(tablesize[1])]

    values = [None]*len(keys)
    for i, key in enumerate(keys):
        col_info = []
        index = []
        for m in range(len(label_list)):
            if key in label_list[m]:
                col_info.append(list_info[m])
                index.append(label_list[m][0])

        values[i] = pd.Series(col_info, index=index)
        values[i] = values[i].to_dict()  # Deduplizierung
        values[i] = pd.Series(values[i])

    dict_info = dict(zip(keys, values))
    # print(dict_info)
    df = pd.DataFrame(dict_info)
    df = df.fillna('(empty_cell)')
    return df


def TableType(df_dict):
    '''
    Beurteilen: Überschrift, Zeilenüberschrift, Subüberschrift, Value

    - input: dict

    - output 1: dict
    - output 2: table_type -- einfach oder nicht

    '''

    header_rN = len(df_dict)//2
    predict_header = [list(dict(pp).values())
                      for pp in list(df_dict.values())[0:header_rN]]
    # hier z.B.
    # [['Project', '2019.12.31', '2020.9.30'],
    #  ['Total Assets', '10,991,903,55', '12,049,642,76']

    if predict_header[0][0] == '(empty_cell)':
        predict_header[0][0] = '#col0row0#'

    # in list will store the index of row who has empty cell, col0 row0 is voll now
    judge_list = []
    for i, row in enumerate(predict_header):
        if '(empty_cell)' in row:
            judge_list.append(i)

    if len(judge_list) == 0:  # die Tabelle ist einfache Tabelle

        table_type = 'einfach'

    else:  # die Tabelle ist komplexe Tabelle
        table_type = 'komplex'

    return table_type


def Einfachverarbeitung(df_dict):
    newDictKeys = list(dict(list(df_dict.values())[0]).values())
    newDictKeys = [pp.replace(' ', '_') for pp in newDictKeys]

    if newDictKeys[0][0] == '(empty_cell)':
        newDictKeys[0][0] = '#col0row0#'

    newDictValues = [None]*len(newDictKeys)
    for n, v in enumerate(newDictValues):
        newDictValues[n] = [None]*(len(df_dict)-1)
        # kann hier nicht newDictValues = [[None]*(len(df_dict)-1)]*len(newDictKeys) schreiben, weil shallow copy gibt es bug

    for i, (key, value) in enumerate(list(df_dict.items())[1:]):
        for ii, (col, info) in enumerate(list(dict(value).items())):
            newDictValues[int(col[3:])][int(key)-1] = info

    newDict = dict(zip(newDictKeys, newDictValues))

    return newDict


def Transform(df_dict):
    '''
    transform dict (infos in row form) to list (infos in columen form)

    '''

    df_list = [None]*len(list(dict(list(df_dict.values())[0]).values()))

    for n, v in enumerate(df_list):
        df_list[n] = [None]*(len(df_dict))
        # kann hier nicht newDictValues = [[None]*(len(df_dict)-1)]*len(newDictKeys) schreiben, weil shallow copy gibt es bug

    for i, (key, value) in enumerate(list(df_dict.items())):
        for ii, (col, info) in enumerate(list(dict(value).items())):
            df_list[int(col[3:])][int(key)] = info

    if df_list[0][0] == '(empty_cell)':
        df_list[0][0] = '#col0row0#'

    return df_list


def VertikalSchmelzen(df_list):

    row_list = []
    number = len(df_list[0])

    for i in range(number):
        row_i = []

        for row in df_list:
            row_i.append(row[i])
        row_list.append(row_i)

    sch_row = []

    for i in range(len(row_list)-1):
        zwei_rows = [row_list[i][m]+row_list[i+1][m] for m in range(len(row_list[i]))]
        result = all(['(empty_cell)' in cell for cell in zwei_rows[2:]])
        if result == True:
            if i-1 not in sch_row:
                sch_row.append(i)

    if len(sch_row) != 0:

        for n in sch_row:

            for spalt in df_list:
                spalt[n] = (spalt[n] + '_' + spalt[n+1]
                              ).replace('_(empty_cell)', '').replace('(empty_cell)_', '').replace('-_', '-')

        df_list_new = [None]*len(df_list)
        for i in range(len(df_list_new)):
            df_list_new[i] = [cell for ii, cell in enumerate(df_list[i]) if ii-1 not in sch_row]
 
        return df_list_new
    else:
        return df_list


def ZeilenIndexSchmelzen(df_list):

    halb_rN = len(df_list[0])//3 + 1

    if '(empty_cell)' in df_list[0][halb_rN:] + df_list[1][halb_rN:]:
        for i, ind in enumerate(df_list[0]):
            df_list[0][i] = (df_list[0][i] + '_' + df_list[1][i]
                             ).replace('_(empty_cell)', '').replace('(empty_cell)_', '')
        del df_list[1]

    return df_list


def BestimmenZeilNummer(df_list):
    '''
    in der Zukunft könnte die Function von ML modell ersetzt werden.

    '''

    row_list = []
    number = len(df_list[0])

    for i in range(number):
        row_i = []

        for row in df_list:
            row_i.append(row[i])
        row_list.append(row_i)

    str_set = ['(empty_cell)']
    voll_row = []
    empty_row = []
    for i, row in enumerate(row_list):
        # if any cell is empty, result = True, d.h. empty row
        result = any([cell in str_set for cell in row])
        if result == True:
            empty_row.append(i)
        else:
            if i != 0 and i != 1:  # Die ersten beiden Zeilen nehmen nicht am Urteil teil
                voll_row.append(i)

    if len(voll_row) != 0:
        zeile_nummer = voll_row[0]
    else:
        zeile_nummer = number//3 + 1

    return zeile_nummer, empty_row, row_list


def HeaderSchmelzen(df_list, zeile_nummer, empty_row, row_list):

    if 0 in empty_row:

        empty_cell = []
        # hier sind die indexs von der Zellen, die aufgefüllt werden muss.
        first_header = []
        # hier sind die indexs von primär header, die coppiert werden muss.

        for col, value_row0 in enumerate(row_list[0][1:]):
            if value_row0 == '(empty_cell)':
                empty_cell.append(col+1)
            else:
                first_header.append(col+1)

        header_zeile = row_list[:zeile_nummer]

        headercol_list = []
        for i in range(len(row_list[0])):
            col_i = []

            for row in header_zeile:
                col_i.append(row[i])
            headercol_list.append(col_i)

        str_set = ['(empty_cell)']
        no_quali = []
        for i, col in enumerate(headercol_list):
            # if all cell are empty, result = True
            result = all([cell in str_set for cell in col[1:]])
            if result == True:
                no_quali.append(i)

        for col in first_header:
            if col+1 not in empty_cell:
                no_quali.append(col)

        first_header = [pri for pri in first_header if pri not in no_quali]
        empty_cell = [emp for emp in empty_cell if emp not in no_quali]

        # The first-level title will dilate to the left and right
        # If a empty cell receives an update request from both the left and the right, the left takes precedence.

        b = 1
        while len(first_header) != 0:
            delete = []

            for col in first_header:
                # expand to the right
                if col+b in empty_cell:
                    headercol_list[col+b][0] = headercol_list[col][0]
                    del empty_cell[empty_cell.index(col+b)]
                else:
                    # basierend auf PositionCorrection ist die rechte Seite unbedingt leer, falls nicht leer ist es ein Fehler.
                    delete.append(col)

            for col in first_header:    
                if col-b in empty_cell:
                    headercol_list[col-b][0] = headercol_list[col][0]
                    del empty_cell[empty_cell.index(col-b)]
                else:
                    # In der vorherigen for-Schleife wurde die leere Zelle auf der rechten Seite der Titelzelle gefüllt.
                    # Wenn die linke Seite zu diesem Zeitpunkt nicht gefüllt werden kann, sollte die Titelzelle deaktiviert werden.
                    delete.append(col)

            delete = list(set(delete))
            for col in delete:
                del first_header[first_header.index(col)]

            b += 1

            if b >= 20:
                break

    else:
        header_zeile = row_list[:zeile_nummer]

        headercol_list = []
        for i in range(len(row_list[0])):
            col_i = []

            for row in header_zeile:
                col_i.append(row[i])
            headercol_list.append(col_i)

    newDictKeys = ['_'.join(item) for item in headercol_list]
    newDictKeys = [pp.replace('(empty_cell)_', '') for pp in newDictKeys]
    newDictKeys = [pp.replace('_(empty_cell)', '') for pp in newDictKeys]
    newDictKeys = [pp.replace('#col0row0#_', '') for pp in newDictKeys]
    newDictKeys = [pp.replace(' ', '_') for pp in newDictKeys]
    newDictKeys = [pp.replace('-', '_') for pp in newDictKeys]
    # in keys should not have  ' ', '-' etc.
    newDictKeys = [pp.replace('.', '_') for pp in newDictKeys]
    newDictKeys = [pp.replace('__', '_') for pp in newDictKeys]
    

    newDictValues = [col[zeile_nummer:] for col in df_list]

    newDict = dict(zip(newDictKeys, newDictValues))

    return newDict


def Umform(df_dict, label_, error_info):
    '''
        see issue: instraction to funciton Umform()

        - input 1: df_dict is a dict, in it is detected table
        - input 2: label_ is the quelle of the table

        - output: processed table information, key is header, value is value in columen form

    '''
    try:
        table_type = TableType(df_dict)

        if table_type == 'einfach':
            df_dict_done = Einfachverarbeitung(df_dict)

            return df_dict_done

        else:
            df_list = Transform(df_dict)
            
            i = 0
            while i < 3:
                df_list = VertikalSchmelzen(df_list)
                i += 1
            
            m = 0
            while m < 2:
                df_list = ZeilenIndexSchmelzen(df_list)
                m += 1

            zeile_nummer, empty_row, row_list = BestimmenZeilNummer(df_list)
            df_dict_done = HeaderSchmelzen(
                df_list, zeile_nummer, empty_row, row_list)

            return df_dict_done

    except Exception as e:
        error_info.append((label_, 'Umform', str(e)))


def WriteData(df, img_path, nummer, error_info):
    '''
        write dataframe to elasticsearch

        - input 1: dataframe
        - input 2: path
        - input 3: table nummer
        - input 4: size of table, also col number and row number

    '''
    try:

        label_ = 'table_' + str(nummer+1) + '_of_' + os.path.basename(img_path)
        df_json = df.to_json(
            orient='index')  # str like {index -> {column -> value}}。
        df_dict = eval(df_json)  # chance str to dict

        df_dict = Umform(df_dict, label_, error_info)

        # einschreibung in elasticsearch mit form in 17.08.2022.md
        df = pd.DataFrame(df_dict)
        df_json = df.to_json(
            orient='index')  # str like {index -> {column -> value}}。
        df = eval(df_json)  # chance str to dict

        values = []
        # actions = []
        for key, value in list(df.items()):
            value = dict(value)

            values.append(value)
        for bulk in values:
            actions = []
            bulk["uniqueId"] = label_.lower()
            bulk["fileName"] = os.path.basename(img_path)

            
            actions.append(bulk)
            helpers.bulk(es, actions, index='table')
    except Exception as e:
        error_info.append(('table_' + str(nummer+1) + '_of_' +
                          os.path.basename(img_path), 'WriteData', str(e)))


def SaveTable(nummer, table, img_path, error_info, model, list_output):
    '''
        This function is the analysis and writing of the table area.
        The purpose of the function is to make multiple tables in the same graph not affect each other.
        The failure of table 1 will not affect the processing of subsequent table 2.

        - input 1: nummer of table
        - input 2: infos of table
        - input 3: is a parameter for WriteData()

        - output: None
    '''
    try:
        table_gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)  # gray image
        table_ol = DeletLines(table_gray)  # bina_image ohne Linien

        location, image_add = GetCell(
            table, table_ol)  # hier subplot(224)

        if __name__ == '__main__':
            plt.suptitle('table ' + str(nummer+1) + ' of ' + str(img_path))
            plt.subplot(131), plt.imshow(table)
            plt.xticks([]), plt.yticks([])
            plt.subplot(132), plt.imshow(table_ol, cmap='gray')
            plt.xticks([]), plt.yticks([])
            plt.subplot(133), plt.imshow(image_add)
            plt.xticks([]), plt.yticks([])
            plt.show()

        col_contours = GetColumn(table, model)

        for col, w in col_contours:
            cv2.line(image_add, (col, 0), (col, 500), color=(0, 0, 255),
                     thickness=1, lineType=cv2.LINE_AA)  # draw the white line on black image
            cv2.line(image_add, (col-w//2, 0), (col-w//2, 1100), color=(0, 255, 0),
                     thickness=1, lineType=cv2.LINE_AA)  # draw the white line on black image
            cv2.line(image_add, (col+w//2, 0), (col+w//2, 1100), color=(0, 255, 0),
                     thickness=1, lineType=cv2.LINE_AA)  # draw the white line on black image
        cv2.imwrite('.\\Development\\imageSave\\{}'.format(
            'table_' + str(nummer+1) + '_of_' + str(os.path.basename(img_path))), image_add)

        center_list, label_list, tablesize = GetLabel(
            location, col_contours)

        list_info = ReadCell(center_list, table)

        df = GetDataframe(list_info, label_list, tablesize)

        if __name__ == '__main__':
            print('--------------------------------------------------')
            print('table %s' % (nummer+1))
            # print(list_info)
            # print(label_list)
            print(df)

        WriteData(df, img_path, nummer, error_info)
        list_output.append('table_' + str(nummer+1) + '_of_' +
                           os.path.basename(img_path))
    except Exception as e:
        error_info.append(('table_' + str(nummer+1) + '_of_' +
                          os.path.basename(img_path), 'SaveTable', str(e)))


def Search(index_, uniqueId):
    '''
    Searches for data in ES-index, for example: table_2_of_xxx.png

    - input 1: index_ is 'table'
    - input 2: uniqueId of table, for example: table_2_of_xxx.png
                if uniqueId is all --> back all datas

    - output: result
    '''
    if uniqueId == 'all':
        reqBody = {
            "size": 1000,  # no. of hits that will be sent
            "query": {
                "match_all": {}  # gives back all entries in ES-index
            }
        }
    else:
        reqBody = {
            "size": 1000,  # No. of hits that will be sent
            "query": {
                "match": {
                    'uniqueId': {
                        'query': uniqueId,
                        'operator': 'and'
                    }
                }
            }
        }

    res = es.search(index=index_, body=reqBody)

    # preperation for pretty-print: encoding with utf-8 for "ä, ö, etc."
    data_print = json.dumps(res, indent=4, ensure_ascii=False).encode('utf8')
    # print(data_print.decode()) # pretty-print with indent level
    return data_print.decode()

#---------------------------------------------------------------------------------------------------------------#


def Main(img_path, model, error_info, list_output):

    try:
        image = cv2.imread(img_path, 0)
        image_rotate = TiltCorrection(image)  # got gray
        # text_zone = WhiteBordersRemove(image_rotate)  # got gray
        img_3channel = cv2.cvtColor(
            image_rotate, cv2.COLOR_GRAY2BGR)  # gray to 3 channel

        img_1024 = SizeNormalize(img_3channel)

        if __name__ == '__main__':
            plt.suptitle('Vorbreitung')
            plt.subplot(131)
            plt.title('Original Image')
            plt.imshow(image, cmap='gray')
            plt.xticks([]), plt.yticks([])
            plt.subplot(132)
            plt.title('Tilt Correction')
            plt.imshow(image_rotate, cmap='gray')
            plt.xticks([]), plt.yticks([])
            # plt.subplot(143)
            # plt.title('White Borders Remove')
            # plt.imshow(img_3channel)
            # plt.xticks([]), plt.yticks([])
            plt.subplot(133)
            plt.title('Resize if not 1024x1024')
            plt.imshow(img_1024)
            plt.xticks([]), plt.yticks([])
            plt.show()
            plt.close()

            # input image must be 3 channel 1024x1024. out img 1024x1024
        table_boundRect = PositionTable(
            img_1024, img_path, model)  # unet besser

        table_zone = GetTableZone(table_boundRect, img_1024)

        # print('image ' + str(img_path) + ' has ' +
        #      str(len(table_zone)) + ' table(s)')

        for nummer, table in enumerate(table_zone):
            SaveTable(nummer, table, img_path, error_info,
                      model, list_output)  # densecol besser

    except Exception as e:
        error_info.append((os.path.basename(img_path), 'Main', str(e)))


if __name__ == '__main__':
    img_path = 'Development\\imageTest\\test6.png'

    es.indices.delete(index='table', ignore=[400, 404])  # deletes whole index

    error_info = []
    list_output = []
    Main(img_path, model='densenet', error_info=error_info, list_output=list_output)
    # model: 'densenet' or 'unet'
    print(error_info)
    
    time.sleep(2)
    results = Search('table', 'all')
    print(results)
    

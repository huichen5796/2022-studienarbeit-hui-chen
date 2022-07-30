from elasticsearch import Elasticsearch
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
import taichi as ti
ti.init()
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

3. WhiteBordersRemove(image):
    remove excess white edges of image
    - input: must be gray image
    - output: gray image with reasonably sized white border white border

4. SizeNormalize(image):
    Normalize the input image size to 1024 x 1024
    - input: image, 1 channel or 3 channel
    - output: image 1024 x 1024

5. PositionTable(img_1024, img_path):
    get the position of table in a image
    - input 1: image must be 3 channel, 1024 x 1024
    - input 2: the path of the image
    - input 3: the used model ---> 'tabelnet', 'densenet' or 'unet'
    - output: the location of tables in image [[x, y, w, h], ..] here x and y are the locaiton of top left point

6. GetTableZone(table_boundRect, img_1024): 
    ROI the table in image
    - input 1: location of tables in image
    - input 2: image
    - output: table_zone

7. DeletLines(img)
    delet all the lines on image
    - input: gray image
    - output: bina image without lines

    at first call the function LSDGetLines() to mark all lines on image
    then in function OrImage() mit cv2.bitwise_or to delet all the lines

8. GetCell(img_deletline):
    Get word blocks by dilate
    - input: bina image without lines
    - output: contour location of text blocks

9. GetLabel(location):
    Assign row and column labels to each cell
    - input: the location of cells (not aligned), [[x,y,w,h], ..] here x and y are the locaiton of top left point of cell
    - output1: the center locationn of cells (aligned), [[center_x, center_y, w, h, x, y], ..]
    - output2: the label of each cell, [[row?, col?], ..]
    - output3: the size of table, [[row_number, col_number], [...], ...]

    during this function the function PointCorrection will be called to align the points

10. ReadCell(center_list, image):
    ORI of each cell and OCR by tesseract
    - input 1: center_list of the table
    - input 2: the image with table
    - output: infos in each cell 

11. GetDataframe(list_info, label_list, tablesize):
    Rebuild the table in a Dataframe
    - input 1: list_info
    - input 2: label_list
    - input 3: tablesize
    - output: Dataframe

12. WriteData(df, label_):
    write dataframe to elasticsearch
    - input 1: dataframe
    - input 2: label_, here is the table name, for example: table_2_of_table2_rotate_0

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
        out = self.upsample_3_table(out)  # [1, 3, 1024, 1024]
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

# encoder-decoder model U-Net


class conv_block(nn.Module):

    def __init__(self, input_channels, output_channels, down=True):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_channels),
                                  nn.ReLU(inplace=True),

                                  nn.Conv2d(output_channels, output_channels,
                                            kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_channels),
                                  nn.ReLU(inplace=True)
                                  )

    def forward(self, x):
        x = self.conv(x)

        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, 32)
        self.Conv2 = conv_block(32, 64)
        self.Conv3 = conv_block(64, 128)

        self.Conv4 = conv_block(128, 256)
        self.Conv5 = conv_block(256, 512)

        self.Up5 = up_conv(512, 256)
        self.Up_conv5 = conv_block(512, 256)

        self.Up4 = up_conv(256, 128)
        self.Up_conv4 = conv_block(256, 128)

        self.Up3 = up_conv(128, 64)
        self.Up_conv3 = conv_block(128, 64)

        self.Up2 = up_conv(64, 32)
        self.Up_conv2 = conv_block(64, 32)

        self.Conv_1x1 = nn.Sequential(nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0),
                                      nn.Sigmoid()
                                      )

    def forward(self, x):

        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
#---------------------------------------------------------------------------------------------------------------#
# functions


def LSDGetLines(img, minLong):
    '''
    lines be marked by LSD 

    - input 1: is a bina image
    - input 2: the min long of lines

    - output 1: is a new black image with same shape of input image, on it is the lines of image, location to location
    - output 2: list of lines, [[x0,y0,x1,y1],[x2,y2,x3,y3],[...],[...],...] 

    '''

    # make a new black image with the same shape of input img
    copy_image = np.zeros((img.shape[0], img.shape[1]))

    lsd = cv2.createLineSegmentDetector(0)
    # get all the location of lines by LSD, if no line, dlines = None
    dlines = lsd.detect(img)[0]
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


def HoughGetLines(img, minLong):  # this function be used now not
    '''
    lines be marked by LSD 

    - input 1: is a bina image
    - input 2: the min long of lines

    - output 1: is a new black image with same shape of input image, on it is the lines of image, location to location
    - output 2: list of lines, [[x0,y0,x1,y1],[x2,y2,x3,y3],[...],[...],...] 

    '''
    # Line makieren durch HoughLines()
    # apertureSize is the size of kernel, also soble
    edges = cv2.Canny(img, 50, 250, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1.0, np.pi/180, 50,
                            minLineLength=minLong, maxLineGap=0)
    copy_image = np.zeros((img.shape[0], img.shape[1]))
    longLines = []
    if lines is not None:
        for line in lines:
            x0, y0, x1, y1 = line[0]
            longLines.append([x0, y0, x1, y1])
            cv2.line(copy_image, (x0, y0), (x1, y1), color=255,
                     thickness=3, lineType=cv2.LINE_AA)

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
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)  # Gaussian binar
    longLines = LSDGetLines(bina_image, minLong=80)[1]

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
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)  # Gaussian binar
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


def TiltCorrection(img):
    '''
    - input: the image we want to tilt correct, must be gray image
    - output: the tilt corrected image

    '''
    angle = GetLineAngle(img)
    if angle == 'nolines':  # if no lines in image, then use GetBoxAngle
        angle = GetBoxAngle(img)

    image_rotate = ImageRotate(img, angle)

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
        gray_image1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)  # Gaussian binar

    bina_image1 = cv2.bitwise_not(bina_image)  # invert the image

    # round all the white pixel by a rect
    x, y, w, h = cv2.boundingRect(bina_image1)
    thickness = 25

    text_zone = np.ones((h+2*thickness, w+2*thickness, 1))

    text_zone = gray_image1[(y-thickness):(y+h+thickness),
                            (x-thickness):(x+w+thickness)]

    return text_zone


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
        path = 'Development\\models\\densenet_100epochs.pkl'
        model = torch.load(path, map_location=torch.device(device))

    elif model_used == 'unet':
        path = 'Development\\models\\unet_model_100epochs.pkl'
        model = torch.load(path, map_location=torch.device(device))

    elif model_used == 'tablenet':
        model = TableNet().to(device)
        path = 'Development\\models\densenet_config_4_model_checkpoint.pth.tar'

        pop_list = ["column_decoder.conv_8_column.0.weight",
                    "column_decoder.conv_8_column.0.bias",
                    "column_decoder.conv_8_column.3.weight",
                    "column_decoder.conv_8_column.3.bias",
                    "column_decoder.upsample_1_column.weight",
                    "column_decoder.upsample_1_column.bias",
                    "column_decoder.upsample_2_column.weight",
                    "column_decoder.upsample_2_column.bias",
                    "column_decoder.upsample_3_column.weight",
                    "column_decoder.upsample_3_column.bias"]
        params = torch.load(path, map_location=torch.device(device))[
            'state_dict']

        for key in pop_list:
            params.pop(key)

        model.load_state_dict(params)

    transform = A.Compose([
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
        pred = model(image)

        if model_used == 'unet':
            pred = (pred.cpu().detach().numpy().squeeze())
        else:
            pred = torch.sigmoid(pred)

            pred = (pred.cpu().detach().numpy().squeeze())

        pred[:][pred[:] > 0.5] = 255.0
        pred[:][pred[:] < 0.5] = 0.0
        pred = pred.astype('uint8')

    # get contours of the prognose to get tables
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
            print('Current Image ==>' + str(img_path))

    table_boundRect = [None]*len(table_contours)
    for i, c in enumerate(table_contours):
        polyline = cv2.approxPolyDP(c, 5, True)
        table_boundRect[i] = cv2.boundingRect(polyline)

    table_boundRect = sorted(table_boundRect, key=lambda x: x[1])
    # draw bounding boxes
    color = (255, 0, 0)  # red

    white_image = np.ones((1024, 1024, 3), np.uint8)*255
    for x, y, w, h in table_boundRect:
        size = 4
        #cv2.rectangle(img, (x,y),(x+w,y+h), color, thickness)
        triangle = np.array(
            [[x-size, y-size], [x-size, y+h+size], [x+w+size, y+h+size], [x+w+size, y-size]])
        cv2.fillConvexPoly(white_image, triangle, color)

    image_add = cv2.addWeighted(img_1024, 0.9, white_image, 0.5, 0)

    if __name__ == '__main__':
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.title('Input Image 1024x1024')
        plt.imshow(img_1024)
        plt.subplot(1, 3, 2)
        plt.title('Output Prognose')
        plt.imshow(pred, cmap='gray')
        plt.subplot(1, 3, 3)
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
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 3)
    img2 = LSDGetLines(img1, minLong=18)[0]

    img_deletline = OrImage(img1, img2)

    return img_deletline


def GetCell(img_deletline):
    '''
    Get word blocks by dilate

    - input: bina image without lines

    - output: contour location of text blocks

    '''

    img_deletline_inv = cv2.bitwise_not(img_deletline)
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    bina_image = cv2.erode(img_deletline_inv, kernel, iterations=1)
    # reduce the noise
    '''

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 7))
    bina_image = cv2.dilate(img_deletline_inv, kernel, iterations=1)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    #bina_image = cv2.erode(bina_image,kernel,iterations = 1)

    ret, bina_image = cv2.threshold(
        bina_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, h = cv2.findContours(
        bina_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # round the text zone by rect
    image_copy = cv2.bitwise_not(img_deletline_inv)
    list_contours = []
    average_cellsize = [0, 0]
    i = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 8:

            list_contours.append((x, y, w, h))
            average_cellsize[0] = average_cellsize[0] + w
            average_cellsize[1] = average_cellsize[1] + h
            i += 1
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), 0,
                          2)  # round the text zone by rect

    average_cellsize = [average_cellsize[0] // i, average_cellsize[1] // i]
    if __name__ == '__main__':
        print('average_cellsize [w, h] --> ' + str(average_cellsize))

    location = np.array(list_contours)

    return location, average_cellsize, image_copy  # image_copy is used only for show


def PointCorrection(location, average_cellsize):
    '''
    align the points

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
    # Disrupted the ordering and caused the region to not be closed
    # Can't get correct cells with intersections that are not aligned, need to correction

    parameter = 1

    location = sorted(location, key=lambda x: x[0])

    for i in range(len(location)-1):
        if location[i+1][0] == location[i][0]:
            continue
        else:
            # suppose there are no two cells with distance less than - in x axis
            if abs(location[i+1][0]-location[i][0]) < int(average_cellsize[0]*parameter):
                location[i+1][0] = location[i][0]
            else:
                continue

    location = sorted(location, key=lambda x: (x[1], x[0]))

    for i in range(len(location)-1):
        if location[i+1][1] == location[i][1]:
            continue
        else:
            # suppose there are no two cells with distance less than - in y axis
            if abs(location[i+1][1]-location[i][1]) < int(average_cellsize[1]*parameter):
                location[i+1][1] = location[i][1]
            else:
                continue

    # von oben nach unten zuerst, dann dabei links nach rechts
    location = sorted(location, key=lambda x: (x[1], x[0]))

    return location


def GetLabel(location, average_cellsize):
    '''
    Assign row and column labels to each cell

    - input: the location of cells (not aligned), [[x,y,w,h], ..] here x and y are the locaiton of top left point of cell

    - output1: the center locationn of cells (aligned), [[center_x, center_y, w, h, x, y], ..]
    - output2: the label of each cell, [[row?, col?], ..]
    - output3: the size of table, [[row_number, col_number], [...], ...]

    '''

    label_list = [None]*len(location)

    # get center of cells
    center_list = [None]*len(location)
    for iii, (x, y, w, h) in enumerate(location):
        center_list[iii] = [x+w//2, y+h//2, w, h, x, y]

    center_list = PointCorrection(center_list, average_cellsize)
    # print(center_list)
    cols_list = list(set([pp[0] for pp in center_list]))  # alle x axis
    cols_list.sort()
    # print(cols_list)
    rows_list = list(set([pp[1] for pp in center_list]))  # alle y axis
    rows_list.sort()
    # print(rows_list)
    tablesize = [len(rows_list), len(cols_list)]

    for i, (c_x, c_y, w, h, x, y) in enumerate(center_list):
        #label_list[i] = ['row%s' % (rows_list.index(c_y))]

        label_list[i] = [int(rows_list.index(c_y))]
        label_list[i].append('col%s' % (cols_list.index(c_x)))

    return center_list, label_list, tablesize


def Extrakt_Tesseract(image_cell):
    '''
    OCR of a image

    - input: image

    - output: str

    '''

    pytesseract.pytesseract.tesseract_cmd = 'D:\\for_tesseract\\tesseract.exe'
    result = pytesseract.image_to_string(image_cell, lang='deu')
    # print(result)

    if '\n' in result:
        result = result.replace('\n', '')
    if result == '':
        result = '----'
    return result


def ReadCell(center_list, image):
    '''
    ORI of each cell and OCR by tesseract

    - input 1: center_list of the table
    - input 2: the image with table

    - output: infos in each cell 

    '''

    size = 3

    list_info = []

    for c_x, c_y, w, h, x, y in center_list:

        cell_zone = np.ones((h+2*size, w+2*size, 1))
        cell_zone = image[(y-size):(y+h+size), (x-size):(x+w+size)]
        cell_zone = cv2.resize(
            cell_zone, (cell_zone.shape[1]*4, cell_zone.shape[0]*4))

        cell = cv2.copyMakeBorder(
            cell_zone, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        # cv2.imshow('',cell)
        # cv2.waitKey()
        # print(cell.shape)
        result = Extrakt_Tesseract(cell)

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
    dict_info = dict(zip(keys, values))
    df = pd.DataFrame(dict_info)
    df = df.fillna('')
    return df


def MergeRow(df_dict):
    '''
    Merge cells across rows

    - input: dict

    - output: dict

    '''

    return df_dict


def Umform(df_dict):
    '''
    Beurteilen: Überschrift, Zeilenüberschrift, Subüberschrift, Value

    - input: dict

    - output: dict
    '''

    # {'0': {'col0': 'Project', 'col1': '2019.12.31', 'col2': '2020.9.30'},
    # '1': {'col0': 'Total Assets', 'col1': '10,991,903,55', 'col2': '12,049,642,76'},
    # '2': {'col0': 'Net Assets', 'col1': '1,044,954,84', 'col2': '1,053,487,14'},
    # '3': {'col0': 'Project', 'col1': '2019.1-12', 'col2': '2020.1-9'},
    # '4': {'col0': 'Operating Revenues', 'col1': '286,039,95', 'col2': '211,058,\\/75'},
    # '5': {'col0': 'Net Profit', 'col1': '105,444, 74', 'col2': '91,193,39'}}

    # let 'index'= value of 'col0' ==> '1' is 'Total Assets'
    # let 'col1' = value of 'col1' in '0' ==> 'col1' in '1' is '2019.12.31'
    # wie tun bei 'Projekt'?

    # {'Total Assets': {'2019.12.31': '10,991,903,55', '2020.9.30': '12,049,642,76'},
    # 'Net Assets': {'2019.12.31': '1,044,954,84', '2020.9.30': '1,053,487,14'},
    # 'Operating Revenues': {'2019.12.31': '286,039,95', '2020.9.30': '211,058,75'},
    # 'Net Profit': {'2019.12.31': '105,444,74', '2020.9.30': '91,193,39'}}

    print(df_dict)
    newDictKeys = [None]*(len(list(df_dict.keys()))-1)
    newSubKeys = list(dict(list(df_dict.values())[0]).values())[1:]

    newDictValues = [None]*(len(newDictKeys))
    for i, (key, value) in enumerate(list(df_dict.items())[1:]):
        newDictKeys[i] = list(dict(value).values())[0]
        newSubValues = list(dict(value).values())[1:]
        newDictValues[i] = dict(zip(newSubKeys, newSubValues))

    newDict = dict(zip(newDictKeys, newDictValues))

    print(newDict)

    return newDict


def WriteData(df, img_path, nummer):
    '''
    write dataframe to elasticsearch

    - input 1: dataframe
    - input 2: path
    - input 3: table nummer

    '''

    label_ = 'table_' + str(nummer+1) + '_of_' + os.path.basename(img_path)
    df_json = df.to_json(
        orient='index')  # str like {index -> {column -> value}}。
    df_dict = eval(df_json)  # chance str to dict

    df_dict = Umform(df_dict)

    body_ = {
        "uniqueId": label_.lower(),
        "fileName": os.path.basename(img_path),
        "content": df_dict
    }

    es.index(index='table', body=body_)


def Search(index_, label_):
    '''
    Searches for data in ES-index, for example: table_2_of_test3_0.png

    - input 1: index_ is 'table'
    - input 2: label of table, for example: table_2_of_table2_rotate_0
                if label_ is all --> back all datas

    - output: result
    '''
    if label_ == 'all':
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
                    label_: {
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


def Main(img_path, model):

    try:
        start = time.time()
        image = cv2.imread(img_path, 0)

        image_rotate = TiltCorrection(image)  # got gray

        text_zone = WhiteBordersRemove(image_rotate)  # got gray

        img_3channel = cv2.cvtColor(
            text_zone, cv2.COLOR_GRAY2BGR)  # gray to 3 channel

        img_1024 = SizeNormalize(img_3channel)

        if __name__ == '__main__':
            plt.suptitle('Vorbreitung')
            plt.subplot(141)
            plt.title('Original Image')
            plt.imshow(image, cmap='gray')
            plt.xticks([]), plt.yticks([])
            plt.subplot(142)
            plt.title('Tilt Correction')
            plt.imshow(image_rotate, cmap='gray')
            plt.xticks([]), plt.yticks([])
            plt.subplot(143)
            plt.title('White Borders Remove')
            plt.imshow(img_3channel)
            plt.xticks([]), plt.yticks([])
            plt.subplot(144)
            plt.title('Resize if not 1024x1024')
            plt.imshow(img_1024)
            plt.xticks([]), plt.yticks([])
            plt.show()
            plt.close()

            # input image must be 3 channel 1024x1024. out img 1024x1024
        table_boundRect = PositionTable(img_1024, img_path, model)

        table_zone = GetTableZone(table_boundRect, img_1024)

        print('image ' + str(img_path) + ' has ' +
              str(len(table_zone)) + ' table(s)')

        for nummer, table in enumerate(table_zone):

            table_gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)  # gray image
            table_ol = DeletLines(table_gray)  # bina_image ohne Linien

            location, average_cellsize, image_copy = GetCell(
                table_ol)  # hier subplot(224)

            if __name__ == '__main__':
                plt.suptitle('table ' + str(nummer+1))
                plt.subplot(2, 2, 1), plt.imshow(img_1024)
                plt.xticks([]), plt.yticks([])
                plt.subplot(2, 2, 2), plt.imshow(table)
                plt.xticks([]), plt.yticks([])
                plt.subplot(2, 2, 3), plt.imshow(table_ol, cmap='gray')
                plt.xticks([]), plt.yticks([])
                plt.subplot(2, 2, 4), plt.imshow(image_copy, cmap='gray')
                plt.xticks([]), plt.yticks([])
                plt.show()
                plt.close()

            center_list, label_list, tablesize = GetLabel(
                location, average_cellsize)

            list_info = ReadCell(center_list, table_ol)

            df = GetDataframe(list_info, label_list, tablesize)

            WriteData(df, img_path, nummer)

            end = time.time()
            print('runtime: %s' % (end - start))

            if __name__ == '__main__':
                print('--------------------------------------------------')
                print('table %s' % (nummer+1))
                # print(list_info)
                # print(label_list)
                print(df)

    except Exception as e:
        print('ERROR: ' + ' ' + str(e) + ' ==> ' + str(img_path))

    else:
        print('successfully done: ' + str(img_path))


if __name__ == '__main__':
    img_path = 'Development\imageTest\\test8.jpg'

    es.indices.delete(index='table', ignore=[400, 404])  # deletes whole index

    Main(img_path, model='tablenet')
    # model: 'tablenet', 'densenet' or 'unet'

    time.sleep(2)
    results = Search('table', 'all')
    print(results)

    # show in dataframe
    results = json.loads(results)
    for result in results['hits']['hits']:
        df = pd.DataFrame(result['_source']['content']).stack().unstack(0)
        print('--------------------')
        table_uniqueId = result['_source']['uniqueId']
        print(table_uniqueId)
        print(df)

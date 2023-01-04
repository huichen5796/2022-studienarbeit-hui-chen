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
    pred = cv2.erode(pred, kernel, iterations=3)
    pred = cv2.dilate(pred, kernel, iterations=2)  # remove small zone

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

    plt.imshow(pred)
    plt.savefig("196.svg", bbox_inches='tight',pad_inches = 0)

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

image_path = 'Development\\imageTest\\deepcol.jpg'
image = cv2.imread(image_path, 0)

img_3channel = cv2.cvtColor(
            image, cv2.COLOR_GRAY2BGR)  # gray to 3 channel

col_contours = GetColumn(img_3channel, 'densenet')

for col, w in col_contours:
    cv2.line(img_3channel, (col, 0), (col, 500), color=(0, 0, 255),
                thickness=1, lineType=cv2.LINE_AA)  # draw the white line on black image
    cv2.line(img_3channel, (col-w//2, 0), (col-w//2, 1100), color=(0, 255, 0),
                thickness=1, lineType=cv2.LINE_AA)  # draw the white line on black image
    cv2.line(img_3channel, (col+w//2, 0), (col+w//2, 1100), color=(0, 255, 0),
                thickness=1, lineType=cv2.LINE_AA)  # draw the white line on black image
cv2.imwrite('.\\Development\\imageSave\\{}'.format(
    'table_' + str(1) + '_of_' + str(os.path.basename(image_path))), img_3channel)
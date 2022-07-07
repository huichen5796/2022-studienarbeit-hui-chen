from Funktionen import TiltCorrection, DeletLines, GetCell, ReadCell, GetLabel, GetDataframe, WriteData, Search
import os
import cv2
import numpy as np
import torch.nn as nn
import torch
import torchvision
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import time
from elasticsearch import Elasticsearch
es = Elasticsearch()


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


def PositionTable(img_1024, img_path):
    device = 'cpu'

    model_used = 'tablenet'

    if model_used == 'densenet':
        path = 'models\\densenet_100.pkl'
        model = torch.load(path, map_location=torch.device(device))

    elif model_used == 'unet':
        path = 'models\\unet_model_100.pkl'
        model = torch.load(path, map_location=torch.device(device))
    elif model_used == 'tablenet':
        model = TableNet().to(device)
        path = 'models\\densenet_config_4_model_checkpoint.pth.tar'

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
        if cv2.contourArea(c) > 1000:
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

    return table_boundRect, img_1024


def WhiteBordersRemove(gray_image):
    # add border for debug of removing
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
    # then normalize the shape to 1024 X ()
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


def GetTableZone(table_boundRect, img_1024):
    table_zone = [None]*len(table_boundRect)
    for ii, (x, y, w, h) in enumerate(table_boundRect):
        t = 50

        table_zone[ii] = np.ones((h, w, 3))

        table_zone[ii] = cv2.copyMakeBorder(img_1024[(y):(y+h), (x):(x+w)], t, t, t, t, cv2.BORDER_CONSTANT, value=(255,255,255))

    return table_zone


def Main(img_path):
    try:
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
        table_boundRect, img_1024 = PositionTable(img_1024, img_path)

        table_zone = GetTableZone(table_boundRect, img_1024)

        print('image ' + str(img_path) + ' has ' +
              str(len(table_zone)) + ' table(s)')

        for nummer, table in enumerate(table_zone):

            table_gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)  # gray image
            table_ol = DeletLines(table_gray)  # bina_image ohne Linien

            location = GetCell(table_ol)  # hier subplot(224)

            if __name__ == '__main__':
                plt.suptitle('table ' + str(nummer+1))
                plt.subplot(2, 2, 1), plt.imshow(img_1024)
                plt.xticks([]), plt.yticks([])
                plt.subplot(2, 2, 2), plt.imshow(table)
                plt.xticks([]), plt.yticks([])
                plt.subplot(2, 2, 3), plt.imshow(table_ol, cmap='gray')
                plt.xticks([]), plt.yticks([])
                plt.show()
                plt.close()

            center_list, label_list, tablesize = GetLabel(location)

            list_info = ReadCell(center_list, table_ol)

            df = GetDataframe(list_info, label_list, tablesize)

            WriteData(df, index_='table_' + str(nummer+1) + '_of_' + os.path.splitext(os.path.basename(img_path))[0])

            if __name__ == '__main__':
                print('--------------------------------------------------')
                print('table %s' % (nummer+1))
                # print(list_info)
                # print(label_list)
                print(df)

                time.sleep(1)
                result = Search('table_' + str(nummer+1) + '_of_' + os.path.splitext(os.path.basename(img_path))[0])
                print(result)

    except Exception as e:
        print('ERROR: ' + ' ' + str(e) + ' ==> ' + str(img_path))

    else:
        print('successfully done: ' + str(img_path))


if __name__ == '__main__':
    img_path = 'Development_tradionell\\imageTest\\table2_rotate.png'

    Main(img_path)

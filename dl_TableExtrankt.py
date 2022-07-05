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

from TableExtract import TiltCorrection, DeletLines, GetCell, HorizonalAlignment, ReadCell, GetInfoDict, WriteData


class DenseNet(nn.Module):
    def __init__(self, pretrained = True, requires_grad = True):
        super(DenseNet, self).__init__()
        denseNet = torchvision.models.densenet121(pretrained=True).features
        self.densenet_out_1 = torch.nn.Sequential()
        self.densenet_out_2 = torch.nn.Sequential()
        self.densenet_out_3 = torch.nn.Sequential()

        for x in range(8):
            self.densenet_out_1.add_module(str(x), denseNet[x])
        for x in range(8,10):
            self.densenet_out_2.add_module(str(x), denseNet[x])
        
        self.densenet_out_3.add_module(str(10), denseNet[10])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        
        out_1 = self.densenet_out_1(x) #torch.Size([1, 256, 64, 64])
        out_2 = self.densenet_out_2(out_1) #torch.Size([1, 512, 32, 32])
        out_3 = self.densenet_out_3(out_2) #torch.Size([1, 1024, 32, 32])
        return out_1, out_2, out_3

class TableDecoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(TableDecoder, self).__init__()
        self.conv_7_table = nn.Conv2d(
                        in_channels = 256,
                        out_channels = 256,
                        kernel_size = kernels[0], 
                        stride = strides[0])
        self.upsample_1_table = nn.ConvTranspose2d(
                        in_channels = 256,
                        out_channels=128,
                        kernel_size = kernels[1],
                        stride = strides[1])
        self.upsample_2_table = nn.ConvTranspose2d(
                        in_channels = 128 + channels[0],
                        out_channels = 256,
                        kernel_size = kernels[2],
                        stride = strides[2])
        self.upsample_3_table = nn.ConvTranspose2d(
                        in_channels = 256 + channels[1],
                        out_channels = 1,
                        kernel_size = kernels[3],
                        stride = strides[3])

    def forward(self, x, pool_3_out, pool_4_out):
        x = self.conv_7_table(x)  #[1, 256, 32, 32]
        out = self.upsample_1_table(x) #[1, 128, 64, 64]
        out = torch.cat((out, pool_4_out), dim=1) #[1, 640, 64, 64]
        out = self.upsample_2_table(out) #[1, 256, 128, 128]
        out = torch.cat((out, pool_3_out), dim=1) #[1, 512, 128, 128]
        out = self.upsample_3_table(out) #[1, 3, 1024, 1024]
        return out

class TableNet(nn.Module):
    def __init__(self,encoder = 'densenet', use_pretrained_model = True, basemodel_requires_grad = True):
        super(TableNet, self).__init__()
        
        self.base_model = DenseNet(pretrained = use_pretrained_model, requires_grad = basemodel_requires_grad)
        self.pool_channels = [512, 256]
        self.in_channels = 1024
        self.kernels = [(1,1), (1,1), (2,2),(16,16)]
        self.strides = [(1,1), (1,1), (2,2),(16,16)]
        
        #common layer
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = self.in_channels, out_channels = 256, kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8))

        self.table_decoder = TableDecoder(self.pool_channels, self.kernels, self.strides)


    def forward(self, x):

        pool_3_out, pool_4_out, pool_5_out = self.base_model(x)
        conv_out = self.conv6(pool_5_out) #[1, 256, 32, 32]
        table_out = self.table_decoder(conv_out, pool_3_out, pool_4_out) #torch.Size([1, 1, 1024, 1024])
        return table_out

#---------------------------------------------------------------------------------------------------------------#

def PositionTable(img, img_path):
    device = 'cpu'
    path = 'Development_DL\\Arbeitbreich_DL\\densenet_100.pkl'
    model = torch.load(path, map_location=torch.device(device))


    transform = A.Compose([
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value = 255,
                    ),
                    ToTensorV2()
                ])


    # The model can only process pictures of 1024x1024 size, 
    # so it is necessary to fill the edges of pictures smaller than this size
    

    h = img.shape[0]
    w = img.shape[1]
    bottom = 1024-h
    right = 1024-w
    img = cv2.copyMakeBorder(img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=(255,255,255))

    img = np.array(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))) 

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(img)

    image = transform(image = img)["image"]
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)   
        pred  = model(image)
        pred = torch.sigmoid(pred)
        pred = (pred.cpu().detach().numpy().squeeze())

        pred[:][pred[:]>0.5]=255.0
        pred[:][pred[:]<0.5]=0.0
        pred = pred.astype('uint8')


    #get contours of the prognose to get tables
    contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    table_contours = []
    #remove bad contours
    for c in contours:
        if cv2.contourArea(c) > 1000:
            table_contours.append(c)

    if len(table_contours) == 0:
        print("No Table detected ==> " + img_path)
    else:
        print('done ==>' + img_path)

    table_boundRect = [None]*len(table_contours)
    for i, c in enumerate(table_contours):
        polyline = cv2.approxPolyDP(c, 5, True)
        table_boundRect[i] = cv2.boundingRect(polyline)

    #draw bounding boxes
    color = (255,255,0)

    white_image = np.ones((1024,1024,3),np.uint8)*255
    for x,y,w,h in table_boundRect:
        #cv2.rectangle(img, (x,y),(x+w,y+h), color, thickness)
        triangle = np.array([[x, y], [x,y+h], [x+w, y+h], [x+w, y]])
        cv2.fillConvexPoly(white_image, triangle, color)
    
    image_add = cv2.addWeighted(img, 0.8, white_image, 0.5, 0)

    plt.subplot(1, 3, 2)
    plt.imshow(pred, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.imshow(image_add )

    plt.show()

    return table_boundRect


def WhiteBordersRomove(gray_image):
    bina_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5) # Gaussian binar
    bina_image1 = cv2.bitwise_not(bina_image) # invert the image
    
    x,y,w,h = cv2.boundingRect(bina_image1) # round all the white pixel by a rect
    thickness = 20
    
    text_zone = np.ones((h+2*thickness, w+2*thickness, 1))

    text_zone = gray_image[(y-thickness):(y+h+thickness),(x-thickness):(x+w+thickness)]

    return text_zone


def Main(img_path):
    image = cv2.imread(img_path, 0)

    plt.subplot(141)
    plt.imshow(image, cmap='gray')

    image_rotate = TiltCorrection(image) # gray
    plt.subplot(142)
    plt.imshow(image_rotate, cmap='gray')

    image_rotate = WhiteBordersRomove(image_rotate)

    image = cv2.cvtColor(image_rotate, cv2.COLOR_GRAY2BGR) # 3 channel
    
    plt.subplot(143)
    plt.imshow(image)

    shape_list = list(image.shape)
    print('image_shape ==> '+ str(image.shape))

    if max(shape_list) > 1024:
        scaling_r = 1024/max(shape_list)
        shape = [int(shape_list[0]*scaling_r), int(shape_list[1]*scaling_r)]
        shape[shape_list.index(max(shape_list))] = int(1024)
        image = cv2.resize(image, shape)
        print('new ==> ' + str(image.shape))

    plt.subplot(144)
    plt.imshow(image)
    plt.show()

    table_boundRect = PositionTable(image, img_path) 
    table_zone = [None]*len(table_boundRect)
    for i, (x,y,w,h) in enumerate(table_boundRect):
        t = 2
    
        table_zone[i] = np.ones((h+2*t, w+2*t, 1))

        table_zone[i] = image_rotate[(y-t):(y+h+t),(x-t):(x+w+t)]
    
    print('image '+ img_path + ' has ' + str(len(table_zone)) +' table(s)')

    for table in table_zone:
        table_ol = DeletLines(table)
        location = GetCell(table_ol)
        location = HorizonalAlignment(location)

        list_info = ReadCell(location, table_ol)
        dict_info = GetInfoDict(list_info)
        WriteData(dict_info)


Main('Development_tradionell\\imageTest\\textandtablewinkel.png')
# Hier ist noch eine einfache Gliederung, 
# die Tabelle im Bild wird gelesen und in die Datenbank geschrieben, 
# aber sie kann noch nicht die Verwandtschaft zweier Tabellen aus demselben Bild widerspiegeln.




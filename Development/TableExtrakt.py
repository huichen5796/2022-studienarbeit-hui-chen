import torch.nn as nn
import torch
import torchvision
import json
import pandas as pd
import os
import fitz
import shutil
from functions import Main, Search
from elasticsearch import Elasticsearch
es = Elasticsearch()

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

def GetImageList(dir_name):
    '''
    get all the filename of images unter a dir

    - input: path of dir

    - output: all the files unter the dir, in form list 

    '''

    try:
        file_list = os.listdir(dir_name)
        # if the input is a path of dir then get all the name of the files unter the dir

        return file_list

    except Exception as e:
        print('ERROR BY GetImageList: ' + str(e))


def PDFRemover(dir_name):
    '''
    remove the PDFs unter dir_name to 'Development\\PDF'

    - input: the path of directory

    - output: None

    '''
    try:
        n = 0
        file_list = GetImageList(dir_name)
        for file in file_list:

            if os.path.splitext(file)[1] in ['.pdf', '.PDF']:
                shutil.copy(dir_name + '\\' + file, 'Development\\PDF')
                os.remove(dir_name + '\\' + file)
                n += 1

            else:
                continue
        print("there are %s PDFs, removed to 'Development\\PDF'." % n)

    except Exception as e:
        print('ERROR BY PDFRemover: ' + str(e))


def PdfToPng(pdf_path, save_path):
    '''
    change all PDFs unter dir to PNG

    - input 1: the path of pdf
    - input 2: the path to save the pdf

    - output: None

    '''
    try:
        doc = fitz.open(pdf_path)
        for pg in range(doc.pageCount):  # pg ist die Seitenummer
            page = doc[pg]
            rotate = int(0)
            zoom_x = 2.0
            zoom_y = 2.0
            trans = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
            pm = page.get_pixmap(matrix=trans, alpha=False)
            pm.save(save_path + '\\' +
                    os.path.splitext(os.path.basename(pdf_path))[0] + '_%s.png' % pg)

        print('%s has %d pages' % (os.path.basename(pdf_path), pg+1))
    except:
        print('ERROR BY PdfToPng OF %s' % (pdf_path))


def ImageReformat(dir):
    '''
    - input: path of dir

    - output: image_list

    hier kann alle Images unter a dir vorverarbeitet werden ----- durch ImageReformat() --- in 'image_list'
    dabei wird alle PDFs in 'Development\PDF' umgezogen, 
    Jede Seite der PDF-Datei wird in ein PNG-Bild konvertiert und hier gespeichert ----- 'Development\imageTest'
    '''
    try:
        PDFRemover(dir)
        pdf_list = GetImageList('Development\\PDF')
        if len(pdf_list) == 0:
            image_list = GetImageList(dir)
            print('There are %s images in total, including the images from the pdfs.' % len(
                image_list))
            print('---------------------------')
            # print(image_list)
            return image_list

        else:
            for pdf in pdf_list:
                pdf_path = 'Development\\PDF' + '\\' + pdf
                save_path = 'Development\\imageTest'
                PdfToPng(pdf_path, save_path)

            image_list = GetImageList(dir)
            print('There are %s images in total, including the images from the pdfs.' % len(
                image_list))
            print('---------------------------')
            # print(image_list)
            return image_list
    except:
        print('ERROR BY ImageReformat')


def StapelVerbreitung(dir, model):

    image_list = GetImageList(dir)
    print('---------------------------')
    print('There are %s images in total. -- include unprocessed Pdfs.' %
          len(image_list))
    image_list = ImageReformat(dir)

    # print(image_list)
    path_images = [os.path.normpath(os.path.join(dir, fn))
                   for fn in image_list]
    for image in path_images:
        Main(image, model)
    print('done')


if __name__ == '__main__':

    es.indices.delete(index='table', ignore=[400, 404])  # deletes whole index

    StapelVerbreitung('Development\\imageTest', model = 'unet')
    # model: 'tablenet', 'densenet' or 'unet'

    results = Search('table', 'all')

    # show in dataframe
    results = json.loads(results)
    for result in results['hits']['hits']:
        df = pd.DataFrame(result['_source']['content']).stack().unstack(0)
        print('--------------------')
        table_label = result['_source']['label']
        print(table_label)
        print(df)

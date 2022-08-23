from tkinter.ttk import Progressbar
import torch.nn as nn
import torch
import torchvision
import json
import pandas as pd
import os
import fitz
import time
import copy
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

                pdf_path = dir_name + '\\' + file
                save_path = dir_name
                PdfToPng(pdf_path, save_path)

                shutil.copy(dir_name + '\\' + file, 'Development\\PDF')
                os.remove(dir_name + '\\' + file)
                n += 1

        print("%s PDFs, removed to 'Development\\PDF'" % n)

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

        print('%s has %d pages' % (os.path.basename(pdf_path), doc.pageCount))

        start = time.perf_counter()

        for pg in range(doc.pageCount):  # pg ist die Seitenummer

            page = doc[pg]
            rotate = int(0)
            zoom_x = 2.0
            zoom_y = 2.0
            trans = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
            pm = page.get_pixmap(matrix=trans, alpha=False)
            pm.save(save_path + '\\' +
                    os.path.splitext(os.path.basename(pdf_path))[0] + '_%s.png' % pg)

            finish = '▓' * (pg+1)
            need_do = '-' * (doc.pageCount-pg-1)
            dur = time.perf_counter() - start

            if pg == doc.pageCount-1:
                print("\r{}/{}|{}{}|{:.2f}s".format((pg+1),
                      doc.pageCount, finish, need_do, dur))
            else:
                print("\r{}/{}|{}{}|{:.2f}s".format((pg+1),
                      doc.pageCount, finish, need_do, dur), end='')

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
        image_list = GetImageList(dir)
        return image_list

    except:
        print('ERROR BY ImageReformat')


def StapelVerbreitung(dir, model, list_output):
    error_info = []
    image_list = GetImageList(dir)

    print('---------------------------')
    print('%s images in total. -- includes unprocessed Pdfs.' %len(image_list))

    image_list = ImageReformat(dir)

    print('%s images in total. -- includes processed Pdfs.' %len(image_list))
    print('---------------------------')

    # print(image_list)
    path_images = [os.path.normpath(os.path.join(dir, fn))
                   for fn in image_list]
    start = time.perf_counter()

    for i, image_path in enumerate(path_images):

        Main(image_path, model, error_info, list_output)

        finish = '▓' * int((i+1)*(50/len(path_images)))
        need_do = '-' * (50-int((i+1)*(50/len(path_images))))
        dur = time.perf_counter() - start

        if i == len(path_images)-1:
            print("\r{}/{}|{}{}|{:.2f}s".format((i+1), len(path_images), finish, need_do, dur) +
                  ' done: ' + os.path.basename(image_path)+' ERROR: %s, finish' % len(error_info), flush=True)
        else:
            print("\r{}/{}|{}{}|{:.2f}s".format((i+1), len(path_images), finish, need_do, dur) +
                  ' done: ' + os.path.basename(image_path)+' ERROR: %s' % len(error_info), end='', flush=True)

    print('ERROR: %s' % error_info)

if __name__ == '__main__':

    es.indices.delete(index='table', ignore=[400, 404])  # deletes whole index

    dir_paths= ['Development\\successControl', 'Development\\imageTest']
    model = 'densenet'

    for dir_path in dir_paths:
        StapelVerbreitung(dir_path, model, list_output=[])
        # model: 'tablenet', 'densenet' or 'unet'

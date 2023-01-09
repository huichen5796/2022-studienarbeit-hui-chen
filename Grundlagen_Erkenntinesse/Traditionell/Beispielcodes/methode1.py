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
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'D:\\for_tesseract\\tesseract.exe'

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


def Umform(df_dict, label, error_info):
    '''
        see issue: instraction to funciton Umform()

        - input 1: df_dict is a dict, in it is detected table
        - input 2: label is the quelle of the table

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
        error_info.append((label, 'Umform', str(e)))

def Extrakt_Tesseract(image_cell):
    '''
    OCR of a image

    - input: image

    - output: str

    '''

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
        # result = StrToNr(result)

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

image_path = 'Development\\imageTest\\json.jpg'
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


list_info = ReadCell(center_list, table)

df = GetDataframe(list_info, label_list, tablesize)

df_json = df.to_json(
            orient='index')  # str like {index -> {column -> value}}。
df_dict = eval(df_json)  # chance str to dict

df_dict = Umform(df_dict, label_, error_info)
''' 
# Das Konvertieren von Zeichenfolgen in Zahlen kann beim Speichern zu Verwirrung 
# führen, da manchmal Ganzzahlen aufgrund der automatischen Zuordnung fälschlicherweise 
# als Floats gespeichert werden.

for i, (key, value) in enumerate(df_dict.items()):
    if i != 0:
        for ii, cell in enumerate(value):
            df_dict[key][ii] = StrToNr(cell)
print(df_dict)
'''
# einschreibung in elasticsearch mit form in 17.08.2022.md
df = pd.DataFrame(df_dict)
print(df)
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
    print(actions)
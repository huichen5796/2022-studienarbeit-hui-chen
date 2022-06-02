import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import pytesseract
import time
from elasticsearch import Elasticsearch
es = Elasticsearch()


'''
**Main functions**

1. NoiseReducter(img)
    - cause now is the image with good quality, this function currently only has a binarization function (Gaussian-threshold)
    - input image
    - output binar image

2. TiltCorrection(image)
    

3. DeletLines()


7. GetInfoDict(list_info)
    input
    - list_info

    return
    - dict_info // JSON ----> Each element in the dict is the content of the corresponding cell
                       the order is from left to right and from top to bottom

8. WriteData(dict_info)
    - write the dict_info in elasticsearch

9. Search(index)
    - search the data in elasticsearch
    

'''


def NoiseReducter(img):
    '''
     - cause now is the image with good quality, this function currently only has a binarization function
     - input path of color image
     - output binar image

    '''

    # Gauss Binar
    bina_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

    return bina_image

def GetAngle(img):
    '''
    input     ---    img
    output    ---    angle   

    '''
    bina_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
    bina_image1 = cv2.bitwise_not(bina_image)
    coords = np.column_stack(np.where(bina_image1 > 0))
    angle = -cv2.minAreaRect(coords)[2]   # this function has three return
                                        # [0] -- center point of recttangle
                                        # [1] -- (Length, width) of rectangle
                                        # [2] -- The rotation angle of the rectangle, 
                                        # from the x-axis counterclockwise to the W (width) angle, the range is (-90,0]   

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle == 0:
        angle = angle
    else:
        if angle < -45:
            angle = (90 + angle)
        # otherwise, just take the inverse of the angle to make
        # it positive
        else:
            angle = angle

    return angle

def ImageRotate(image, angle):
    '''
    input   ---  image: the image we want to rotate
    input   ---  angle: rotate anglel
    return  ---  the rotated image

    '''

    # get the shape of the image and then determine the rotate center

    h, w = image.shape[0:2]
    center_X, center_Y = w // 2, h // 2

    # make the rotation matrix
    M = cv2.getRotationMatrix2D((center_X, center_Y), angle, 1.0)

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
    input  --- the image we want to tilt correct, must be gray image
    return --- the tilt corrected image

    '''

    angle = GetAngle(img)
    print(angle)
    image_rotate = ImageRotate(img, angle)
    

    return image_rotate

def LSDGetLines(img):
    '''
    - lines mark by LSD 
    - input is a gray image
    - output is a new white image with same shape of input image, on it is the lines of image, location to location

    input - img             --- binar image
    return - white_image    --- is a bina image

    '''
    long_size = 20
    copy_image = np.zeros((img.shape[0], img.shape[1]))

    lsd = cv2.createLineSegmentDetector(0, scale=1)
    dlines = lsd.detect(img)

    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        long = (y1-y0)*(y1-y0)+(x1-x0)*(x1-x0)
        if long >= long_size*long_size:
            cv2.line(copy_image, (x0, y0), (x1, y1), color=255,
                        thickness=3, lineType=cv2.LINE_AA)
            
    return  copy_image

def OrImage(img1, img2):  
    '''
    add two images, weiss + weiss = weiss, weiss + schwarz = weiss, schwarz +schwarz = schwarz

    '''
    img1 = np.array(img1, np.uint8)
    img2 = np.array(img2, np.uint8)

    image = cv2.bitwise_or(img1, img2)

    return image

def DeletLines(img):
    img1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
    img2 = LSDGetLines(img)
    img_deletline = OrImage(img1, img2)
    

    return img_deletline

def GetCell(img_deletline):
    img_deletline_inv = cv2.bitwise_not(img_deletline)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    bina_image = cv2.erode(img_deletline_inv,kernel,iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    bina_image = cv2.dilate(img_deletline_inv,kernel,iterations = 1)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    #bina_image = cv2.erode(bina_image,kernel,iterations = 1)

    ret, bina_image = cv2.threshold(bina_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, h = cv2.findContours(bina_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    list_contours = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w>20 and h>20:
            
            list_contours.append((x,y,w,h))
            cv2.rectangle(img_deletline, (x,y), (x+w,y+h), 0, 2)
    plt.subplot(2, 2, 4), plt.imshow(img_deletline, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()

    arr_contours = np.array(list_contours)

    return arr_contours

def HorizonalAlignment(location):
    
    location = sorted(location, key=lambda x: x[1])

    for i in range(len(location)-1):
        if location[i+1][1] == location[i][1]:
            continue
        else:
            # suppose there are no cells with height less than 10
            if abs(location[i+1][1]-location[i][1]) < 10:
                location[i+1][1] = location[i][1]
            else:
                continue


    location = sorted(location, key=lambda x: (x[1], x[0]))

    return location


def Extrakt_Tesseract(image_cell):

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    result = pytesseract.image_to_string(image_cell)
    # print(result)

    if '\n' in result:
        result = result.replace('\n', '')
    return result


def ReadCell(location, image_rotate_cor):
    
    list_info = [[]]
    x1,y1,w1,h1 = location[0]
    cell_zone = np.ones((h1, w1, 1))
    cell_zone = image_rotate_cor[(y1):(y1+h1), (x1):(x1+w1)]
    #cell_zone = cv2.adaptiveThreshold(cell_zone, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
    cell_zone = NoiseReducter(cell_zone)

    result = Extrakt_Tesseract(cell_zone)

    list_info[-1].append(result)

    cell_number = 2
    for i in range(1, len(location)):
        if location[i][1] == location[i-1][1]:
            
            x,y,w,h = location[i]

            cell_zone = np.ones((h, w, 1))
            cell_zone = image_rotate_cor[(y):(y+h), (x):(x+w)]
            #cell_zone = cv2.adaptiveThreshold(cell_zone, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
            cell_zone = NoiseReducter(cell_zone)

            result = Extrakt_Tesseract(cell_zone)

            list_info[-1].append(result)

            cell_number += 1

        else:
            list_info.append([])
            x,y,w,h = location[i]

            cell_zone = np.ones((h, w, 1))
            cell_zone = image_rotate_cor[(y):(y+h), (x):(x+w)]
            #cell_zone = cv2.adaptiveThreshold(cell_zone, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
            cell_zone = NoiseReducter(cell_zone)

            result = Extrakt_Tesseract(cell_zone)

            list_info[-1].append(result)

            cell_number += 1
        

    return list_info


def GetInfoDict(list_info):
    '''
    input
     - list_info

    return
     - dict_info // JSON ----> Each element in the dict is the content of the corresponding cell
                       the order is from left to right and from top to bottom

    '''

    key_list = []
    for i in range(len(list_info)):
        key_list.append('row%s' % (i+1))
    dict_info = dict(zip(key_list, list_info))

    if __name__ == '__main__':
        print(dict_info)

    return dict_info


def WriteData(dict_info):
    i = 1
    es.index(index='table', doc_type='_doc', body=dict_info)


def Search(index):
    """ 
    Searches for data in ES-index
    op: operator can be "and" (e.g. must match "Husten AND Fieber") or "or" (e.g. "Husten OR Fieber")
    """

    reqBody = {
        "size": 1000,  # no. of hits that will be sent
        "query": {
            "match_all": {}  # gives back all entries in ES-index
        }
    }

    res = es.search(index=index, body=reqBody)

    # preperation for pretty-print: encoding with utf-8 for "ä, ö, etc."
    data_print = json.dumps(res, indent=4, ensure_ascii=False).encode('utf8')
    # print(data_print.decode()) # pretty-print with indent level
    return data_print.decode()


def TableExtract(path):

    image = cv2.imread(path, 0)
    plt.subplot(2, 2, 1), plt.imshow(image, 'gray')
    plt.xticks([]), plt.yticks([])
    image_rotate = TiltCorrection(image)
    plt.subplot(2, 2, 2), plt.imshow(image_rotate, cmap='gray')
    plt.xticks([]), plt.yticks([])


    image_rotate = DeletLines(image_rotate)
    plt.subplot(2, 2, 3), plt.imshow(image_rotate, cmap='gray')
    plt.xticks([]), plt.yticks([])
    
    location = GetCell(image_rotate)
    location = HorizonalAlignment(location)
    
    list_info = ReadCell(location, image_rotate)
    dict_info = GetInfoDict(list_info)
    WriteData(dict_info)


if __name__ == '__main__':
    es.indices.delete(index='table', ignore=[400, 404])  # deletes whole index

    TableExtract('Development_tradionell\\imageTest\\tabelle_ohne_linien.png')
    time.sleep(1)
    print(Search('table'))

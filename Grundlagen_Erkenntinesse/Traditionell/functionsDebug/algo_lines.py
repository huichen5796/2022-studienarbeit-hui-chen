# https://www.csdn.net/tags/MtTaEg1sMDUxNDc4LWJsb2cO0O0O.html --> LSD

# https://blog.csdn.net/guoyunfei20/article/details/78754526 --> FLD

import cv2
import numpy as np
import time

from torch import long

def LSD(bina_image, long_size):
    start = time.time()
    white = np.ones((bina_image.shape[0], bina_image.shape[1]))
    lsd = cv2.createLineSegmentDetector(0, scale=1)
    dlines = lsd.detect(bina_image)[0]

    for dline in dlines:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))

        long = (y1-y0)*(y1-y0)+(x1-x0)*(x1-x0)
        if long >= long_size*long_size:
            cv2.line(white, (x0, y0), (x1, y1), 0, 1, cv2.LINE_AA)
    end = time.time()
    print('LSD: ' + str(end-start))
    cv2.imshow('LSD', white)

def Hough(bina_image, long_size):
    start = time.time()
    ### Line makieren durch HoughLines()
    edges = cv2.Canny(bina_image, 50, 250, apertureSize= 3) ## apertureSize is the size of kernel, also soble

    lines = cv2.HoughLinesP(edges, 1.0, np.pi/180, 50, minLineLength = long_size, maxLineGap=1) 
    # https://blog.csdn.net/dcrmg/article/details/78880046 # 

    ### show the line
    # img_color = cv2.imread(path, -1)
    '''
    color_img = np.expand_dims(bina_img, axis=2)
    color_img = np.concatenate((color_img, color_img, color_img), axis=-1) 
    
    # change the gray image to color image
    '''
    ### oder: ###
    white = np.ones((bina_image.shape[0], bina_image.shape[1]))

    for line in lines:
        x1, y1, x2, y2 = line[0]

        cv2.line(white, (x1,y1), (x2, y2), 0, 1)

    end = time.time()
    print('Hough: ' + str(end-start))
    cv2.imshow('Hough', white)

def FLD(bina_image, long_size):
    start = time.time()
    white = np.ones((bina_image.shape[0], bina_image.shape[1]))
    fld = cv2.ximgproc.createFastLineDetector()
    dlines = fld.detect(bina_image)

    for dline in dlines:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))

        long = (y1-y0)*(y1-y0)+(x1-x0)*(x1-x0)
        if long >= long_size*long_size:
            cv2.line(white, (x0, y0), (x1, y1), 0, 1, cv2.LINE_AA)

    end = time.time()
    print('FLD: ' + str(end-start))
    cv2.imshow('FLD', white)


img_path = 'Development\\imageTest\\test14.png'
img = cv2.imread(img_path, 0)
bina_image = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

long_size = 15
cv2.imshow('Image', bina_image)
Hough(bina_image, long_size)
LSD(bina_image, long_size)
FLD(bina_image, long_size)
cv2.waitKey()
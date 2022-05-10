## für Kantenerkennung 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pro_processing import Circle2


def Katen(path):
    ### Kantenerkennung durch soble
    bina_img = Circle2(path, 5)
    edges = cv2.Canny(bina_img, 50, 250, apertureSize= 3) ## apertureSize is the size of kernel, also soble

    ### Line makieren durch HoughLines()

    lines = cv2.HoughLinesP(edges, 1.0, np.pi/180, 50, minLineLength=50, maxLineGap=10) 
    # https://blog.csdn.net/dcrmg/article/details/78880046 # 

    ### show the line
    # img_color = cv2.imread(path, -1)
    '''
    color_img = np.expand_dims(bina_img, axis=2)
    color_img = np.concatenate((color_img, color_img, color_img), axis=-1) 
    
    # change the gray image to color image
    '''
    ### oder: ###
    color_img = cv2.merge((bina_img, bina_img, bina_img))

    for line in lines:
        x1, y1, x2, y2 = line[0]

        img_line = cv2.line(color_img, (x1,y1), (x2, y2), (0, 0, 255), 2)

    # cv2.imshow('input', bina_img)
    cv2.imshow("output", img_line)
    cv2.waitKey()

    return img_line

Katen('Development\imageTest\image1.png')






## für Kantenerkennung 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Binarization import Binar


def Katen(path):
    ### Kantenerkennung durch soble
    bina_img = Binar(path)
    edges = cv2.Canny(bina_img, 50, 250, apertureSize= 3) ## apertureSize is the size of kernel, also soble

    ### Line makieren durch HoughLines()

    lines = cv2.HoughLinesP(edges, 1.0, np.pi/180, 50, minLineLength=50, maxLineGap=10) 
    # https://blog.csdn.net/dcrmg/article/details/78880046 # 

    ### show the line
    img_color = cv2.imread(path, -1)

    for line in lines:
        x1, y1, x2, y2 = line[0]

        img_line = cv2.line(img_color, (x1,y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('input', bina_img)
    cv2.imshow("output", img_line)
    cv2.waitKey()

    return img_line

# Katen('Development\imageTest\image1.png')






## https://blog.csdn.net/wsh596823919/article/details/79982485  
## https://baijiahao.baidu.com/s?id=1669080996142912937&wfr=spider&for=pc ##

## hier werden vier verschiedene Blurmethode verglichen.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from Binarization import Binar

def Blur(bina_img):
    img_fil = cv2.blur(bina_img, (5,5))
    
    # cv2.imshow('g_f', img_fil)
    # cv2.waitKey()

    return img_fil

def Box_Filter(bina_img):
    img_fil = cv2.boxFilter(bina_img, -1, (5,5), normalize = 1)

    # cv2.imshow('g_f', img_fil)
    # cv2.waitKey()

    return img_fil

def Gau_Filter(bina_img):
    img_fil = cv2.GaussianBlur(bina_img, (5,5), 0)
    
    # cv2.imshow('g_f', img_fil)
    # cv2.waitKey()

    return img_fil

def Median_Filter(bina_img):
    img_fil = cv2.medianBlur(bina_img, 5)

    # cv2.imshow('g_f', img_fil)
    # cv2.waitKey()

    return img_fil


def Show(path):
    bina_img = Binar(path)
    titles = ['bina_img', 'Blur', 'Box_Filter', 'Gau_Filter', 'Median_Filter']
    images = [bina_img, Blur(bina_img), Box_Filter(bina_img), Gau_Filter(bina_img), Median_Filter(bina_img)]
    for i in range(5):
        plt.subplot(1,5,i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()


Show('Development\imageTest\image2.png')


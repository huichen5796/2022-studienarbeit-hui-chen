## https://blog.csdn.net/wsh596823919/article/details/79982485  
## https://baijiahao.baidu.com/s?id=1669080996142912937&wfr=spider&for=pc ##

## hier werden vier verschiedene Blurmethode verglichen.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from Binarization import Binar

def Blur(bina_img):
    img_fil = cv2.blur(bina_img, (3,3))
    
    # cv2.imshow('g_f', img_fil)
    # cv2.waitKey()

    return img_fil

def Box_Filter(bina_img):
    img_fil = cv2.boxFilter(bina_img, -1, (3,3), normalize = 1)

    # cv2.imshow('g_f', img_fil)
    # cv2.waitKey()

    return img_fil

def Gau_Filter(bina_img):
    img_fil = cv2.GaussianBlur(bina_img, (3,3), 0)
    
    # cv2.imshow('g_f', img_fil)
    # cv2.waitKey()

    return img_fil

def Median_Filter(bina_img):
    img_fil = cv2.medianBlur(bina_img, 3)

    # cv2.imshow('g_f', img_fil)
    # cv2.waitKey()

    return img_fil


def Show(path): # die binarisierte Bilde sind anbei blurred
    bina_img = Binar(path)
    # print(bina_img)
    titles = ['bina_img', 'Blur', 'Box_Filter', 'Gau_Filter', 'Median_Filter']
    i = 0
    
    for image in bina_img:
        images = [image, Blur(image), Box_Filter(image), Gau_Filter(image), Median_Filter(image)]
        zeile_of_images = len(bina_img)

    
        plt.subplot(zeile_of_images, 5, 5*zeile_of_images)


        while i < zeile_of_images:
            for spalt in range(5):
                plt.subplot(zeile_of_images, 5, 5*i+spalt+1), plt.imshow(images[spalt], 'gray')
                if i == 0:
                    plt.title(titles[spalt])
                
                plt.xticks([]),plt.yticks([])
            break
        i+=1

    plt.show()

if __name__ == '__main__':
    Show('Development\\imageTest')





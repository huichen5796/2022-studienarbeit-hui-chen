'''
noch nicht fertig hier
- markieren linien durch HoughLinesP
- get location of table --- not done
# https://wenku.baidu.com/view/8bdffcf175a20029bd64783e0912a21614797fd3.html

'''
import cv2
import numpy as np
import math
from distutils.command.config import LANG_EXT
from tilt_correction import TiltCorrection
from get_linien import LSDGetLines


def LineRow(bina_image):  # get image only with row lines - get horizonal lines

    
    h, w = bina_image.shape
    hori_k = int(math.sqrt(w)*1.2)
                                        # hier für die Kernsize 
                                        # https://blog.csdn.net/weixin_41189525/article/details/121889157
    kernel_hori = cv2.getStructuringElement(cv2.MORPH_RECT, (hori_k, 1))

    image_row = ~cv2.dilate(~bina_image, kernel_hori, iterations=1)  # white zone horizonal dilate then inversion
                                                                     # white lines on black background now
    image_row = cv2.dilate(image_row, kernel_hori, iterations=1)  # restore the line long
                                                                    
    
    if __name__ == '__main__':
        cv2.imshow('Horizonale', image_row)
        cv2.waitKey()
    

    return image_row

def LineCol(bina_image):  # get image only with col lines - get vertikal lines
    
    h, w = bina_image.shape
        
    vert_k = int(math.sqrt(h)*1.2)  

    kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_k))
    image_col = ~cv2.dilate(~bina_image, kernel_vert, iterations=1)

    image_col = cv2.dilate(image_col, kernel_vert, iterations=1)
        
    
    if __name__ == '__main__':
        cv2.imshow('Vertikale', image_col)
        cv2.waitKey()
    
       
    return image_col


def Or_Border(img1, img2): # useless function, just to show the table
    '''
    merge two images

    '''
    borders = cv2.bitwise_or(img1, img2)

    if __name__ == '__main__':
        cv2.imshow('Border', borders)
        cv2.waitKey()

    return borders


def And_Border(img1, img2): # useless function, just to show the table
    '''
    merge two images

    '''
    points = cv2.bitwise_and(img1, img2)

    if __name__ == '__main__':
        cv2.imshow('Border', points)
        cv2.waitKey()

    return points


def GetPoint(path):

    image_rotate_cor = TiltCorrection(path)
    white_image = LSDGetLines(image_rotate_cor, 20)[1]

    white_image = white_image.astype(np.uint8)

    ret, bina_image = cv2.threshold(~white_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if __name__ == '__main__':
        cv2.imshow('', bina_image) 
        cv2.waitKey()
        
    image_row = LineRow(bina_image)
    image_col = LineCol(bina_image)

    #Border(image_row, image_col)
    And_Border(image_row, image_col)


if __name__ == '__main__':
    GetPoint(r'Development\imageTest\einfach_table.jpg')









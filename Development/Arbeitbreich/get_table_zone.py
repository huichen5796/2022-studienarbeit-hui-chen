'''
- binarization
- enhancement
- mark Linien

'''

import cv2
from tilt_correction import TiltCorrection
import math




def GetHorizonale(path, p):  # Entfernen Text und Vertikalen
    bina_image = TiltCorrection(path)
    
    if p == 'horizonal':
        h, w = bina_image.shape
        hors_k = int(math.sqrt(w)*1.2)
                                        # hier für die Kernsize 
                                        # https://blog.csdn.net/weixin_41189525/article/details/121889157
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (hors_k, 1))
        img_hors = ~cv2.dilate(bina_image, kernel1, iterations=1)
        cv2.imshow('Horizonale', img_hors)
        cv2.waitKey()
        return img_hors

    elif p == 'vertikal':
        h, w = bina_image.shape
        
        vert_k = int(math.sqrt(h)*1.2)  # hier für die Kernsize 
                                        # https://blog.csdn.net/weixin_41189525/article/details/121889157
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_k))
        img_vert = ~cv2.dilate(bina_image, kernel1, iterations=1)
        cv2.imshow('Horizonale', img_vert)
        cv2.waitKey()
        return img_vert


def Thicken(img): # LinienVerdickung durch Dilate

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_t = ~cv2.dilate(img, kernel1, iterations=1)

 
    cv2.imshow('verdickte Linien', img_t)
    cv2.waitKey()

    ret,thresh1 = cv2.threshold(img_t, 254, 255, cv2.THRESH_BINARY)
    img_t = ~thresh1
    cv2.imshow('verdickte Linien', img_t)
    cv2.waitKey()

    return img_t

def LinienReparat(img_t, p): # die Linie horizontal verlängern
    if p == 'horizonal':
        h, w = img_t.shape
        hors_k = int(math.sqrt(w)*1.2)
        # vert_k = int(math.sqrt(h)*1.2)  # hier für die Kernsize 
                                        # https://blog.csdn.net/weixin_41189525/article/details/121889157

        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (hors_k, 1))
        img_r = cv2.dilate(img_t, kernel1, iterations=2)

    
        cv2.imshow('Horizonale', img_r)
        cv2.waitKey()
        return img_r

    elif p == 'vertikal':
        h, w = img_t.shape
        
        vert_k = int(math.sqrt(h)*1.2)  # hier für die Kernsize 
                                        # https://blog.csdn.net/weixin_41189525/article/details/121889157

        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_k))
        img_r = cv2.dilate(img_t, kernel1, iterations=2)

    
        cv2.imshow('Horizonale', img_r)
        cv2.waitKey()
        return img_r



########################### Main Funktion ################################

def LinienMakieren(path,p):
    '''
    - path --- the path of image, must after titel correction
    - p --- parameter for (Horizonalen makieren) or (Vertikalen makieren)
                             (p = 'horizonal')         (p = 'vertikal')
    - return img_r --- the image only with 'Horizonalen' or 'Vertikalen'

    '''
    img = GetHorizonale(path, p)
    img_t = Thicken(img)
    img_r = LinienReparat(img_t, p)

    return img_r


def Border(img1, img2):
    '''
    merge two images

    '''
    borders = cv2.bitwise_or(img1, img2)

    cv2.imshow('Border', borders)
    cv2.waitKey()

    return borders


if __name__ == '__main__':
    img1 = LinienMakieren(r'Development\imageTest\rotate_table.png', 'horizonal')
    img2 = LinienMakieren(r'Development\imageTest\rotate_table.png', 'vertikal')
    Border(img1, img2)


##########################################################################

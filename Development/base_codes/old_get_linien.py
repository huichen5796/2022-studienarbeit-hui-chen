
'''
- tilt correction
- enhancement
  - get horizonals 
    - white zone horizonal dilate ---> remove words and vertikals
      ## kernelsize ---> (hors_k, 1)
    - black and white inversion then white vertikal dilate ---> thicken the horizonal lines
    - restore line length ---> dilate by kernelsize ---> (hors_k, 1)
  - get vertikals 
    - white zone vertikal dilate ---> remove words and horizonals
      ## kernelsize ---> (1, vert_k)
    - black and white inversion then white horizonal dilate ---> thicken the horizonal lines
    - restore line length ---> dialte by kernelsize ---> (1, vert_k)
    
- get Linien

'''

import cv2
from tilt_correction import TiltCorrection
import math
import numpy as np




def RemoveLine(path, p):  # Entfernen Text und Vertikalen
    bina_image = TiltCorrection(path)
    #bina_image = cv2.adaptiveThreshold(bina_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
    
    if p == 'horizonal':
        h, w = bina_image.shape
        hors_k = int(math.sqrt(w)*1.2)
                                        # hier für die Kernsize 
                                        # https://blog.csdn.net/weixin_41189525/article/details/121889157
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (hors_k, 1))
        # kernel1 = np.ones((1, hors_k))
        img_hors = ~cv2.dilate(bina_image, kernel1, iterations=1)  # white zone horizonal dilate then inversion
                                                                   # white lines on black background now
                                                                   # img_hors is gray image?
        '''
        if __name__ == '__main__':
            cv2.imshow('Horizonale', img_hors)
            cv2.waitKey()
        '''

        return img_hors

    elif p == 'vertikal':
        h, w = bina_image.shape
        
        vert_k = int(math.sqrt(h)*1.2)  

        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_k))
        img_vert = ~cv2.dilate(bina_image, kernel1, iterations=1)
        
        '''
        if __name__ == '__main__':
            cv2.imshow('Vertikale', img_vert)
            cv2.waitKey()
        '''
       
        return img_vert


def Thicken(img_l, p): # LinienVerdickung durch Dilate
    if p == 'horizonal':
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        # kernel1 = np.ones((2,1))
        img_t = cv2.dilate(img_l, kernel1, iterations=0)
        
        '''
        if __name__ == '__main__':
            cv2.imshow('verdickte Linien', img_t)
            cv2.waitKey()
        '''
    
    elif p == 'vertikal':
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        img_t = cv2.dilate(img_l, kernel1, iterations=0)
        
        '''
        if __name__ == '__main__':
            cv2.imshow('verdickte Linien', img_t)
            cv2.waitKey()
        '''


    return img_t

def LinienRestore(img_t, p): # restore line length
    if p == 'horizonal':
        h, w = img_t.shape
        hors_k = int(math.sqrt(w)*1.2)
        # vert_k = int(math.sqrt(h)*1.2)  # hier für die Kernsize 
                                        # https://blog.csdn.net/weixin_41189525/article/details/121889157

        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (hors_k, 1))
        img_r = cv2.dilate(img_t, kernel1, iterations=1)
        
        '''
        if __name__ == '__main__':
            cv2.imshow('Horizonale restore', img_r)
            cv2.waitKey()
        '''
        return img_r

    elif p == 'vertikal':
        h, w = img_t.shape
        
        vert_k = int(math.sqrt(h)*1.2)  

        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_k))
        img_r = cv2.dilate(img_t, kernel1, iterations=1)
        
        '''
        if __name__ == '__main__':
            cv2.imshow('Vertikale restore', img_r)
            cv2.waitKey()
        '''
        return img_r



def GetLine(path,p):
    '''
    - path --- the path of image, must after titel correction
    - p --- parameter for (Horizonalen makieren) or (Vertikalen makieren)
                             (p = 'horizonal')         (p = 'vertikal')
    - return img_r --- the image only with 'Horizonalen' or 'Vertikalen'

    '''
    img = RemoveLine(path, p)
    img_t = Thicken(img, p)
    img_r = LinienRestore(img_t, p)
    

    return img_r


def Border(img1, img2):
    '''
    merge two images

    '''
    borders = ~cv2.bitwise_or(img1, img2)

    if __name__ == '__main__':
        cv2.imshow('Border', borders)
        cv2.waitKey()

    return borders

def GetTable(path):
    img1 = GetLine(path, 'horizonal')
    img2 = GetLine(path, 'vertikal')
    borders_image = Border(img1, img2)
    return borders_image


if __name__ == '__main__':
    GetTable(r'Development\imageTest\einfach_table.jpg')
    # GetTable(r'Development\imageTest\rotate_table.png')





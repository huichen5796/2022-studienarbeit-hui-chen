import cv2
import numpy as np
import matplotlib.pyplot as plt

import math


# get the shape of the image and then determine the center
# https://blog.csdn.net/qq_44109682/article/details/117434461
def Rotate(path, angle):

    '''
    - path
    - angle:
        # 60 means rotate sixty degrees clockwise
        # -60 means rotate sixty degrees counterclockwise
    - return --- image_rotate

    '''


    image = cv2.imread(path, 0)
    angle = -angle


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
    image_rotate = cv2.warpAffine(image, M, (new_w, new_h),borderValue=(255,255,255))
    # borderValue default（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))


    cv2.imshow('',image_rotate)
    cv2.waitKey()
    cv2.imwrite('winkel_%s.png'%angle, image_rotate)

    return image_rotate

if __name__ == '__main__':
    Rotate(r'Development\imageTest\winkel_0.png', 60)

    # 60 means rotate sixty degrees clockwise
    # -60 means rotate sixty degrees counterclockwise
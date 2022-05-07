## Dieses Programm dient zur horizontalen oder vertikalen Kantenerkennung durch Soble. ### 


import cv2
import numpy as np
from Binarization import Binar


k_soble_horizonal = np.array((
                   [-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]
))

k_soble_vertical = np.array((
                   [-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  0]
))



def Covonlution(img, kernel):

    dst = cv2.filter2D(img, -1, kernel)
    htich = np.hstack((img, dst))
    # cv2.imwrite("test_cov.jpg", htich)
    cv2.imshow('conv', htich)
    cv2.waitKey()







bina_img = Binar('Development\imageTest\image2.png')

Covonlution(bina_img, k_soble_horizonal)
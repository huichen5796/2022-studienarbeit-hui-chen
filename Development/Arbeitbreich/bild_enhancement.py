'''
- binarization
- enhancement

'''

import cv2


def GaussBinar(path):
    gray_image = cv2.imread(path, 0)
    bina_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)

    return bina_image
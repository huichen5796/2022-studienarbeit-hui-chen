
import cv2
import math


def Dilate(bina_image):
    
    h, w = bina_image.shape
    hors_k = int(math.sqrt(w)*1.2)
    # vert_k = int(math.sqrt(h)*1.2)  # hier für die Kernsize 
                                      # https://blog.csdn.net/weixin_41189525/article/details/121889157

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (hors_k, 1))
    hors = ~cv2.dilate(bina_image, kernel1, iterations=1)

 
    cv2.imshow('Horizonale', hors)
    cv2.waitKey()

    return hors
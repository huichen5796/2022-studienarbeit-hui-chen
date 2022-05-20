'''
noch nicht fertig hier
- ROI of Zelle in table
- erode then dilate ---> reduce the noise
   
    ----------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    bina_image = ~cv2.erode(~bina_image, kernel,iterations = 1)   # erode to noise reduction
    # erode should after ROI before Tesseract
    ----------------
    
- Get Words in Zelle
    - Methode 1. Textfokus: https://wenku.baidu.com/view/8bdffcf175a20029bd64783e0912a21614797fd3.html
    - Methode 2. Tesseract
- write Words in JISON
- JISON in Excal


'''

import pytesseract
import cv2
import numpy as np
import pandas as pd
from tilt_correction import TiltCorrection
from get_ROI_location import GetPoint


image_rotate_cor, white_image_cor = TiltCorrection(r'Development\imageTest\einfach_table.jpg')
location = GetPoint(white_image_cor)






def GetTable(img, location, edge_thickness):
    sum_row = list(np.sum(location, axis = 1)) # sum the x-axis and y-axis of point
    # print(location)
    # print(sum_row)
    max_loca = sum_row.index(max(sum_row))
    min_loca = sum_row.index(min(sum_row))
    max_point = location[max_loca]
    min_point = location[min_loca]

    edge_thickness = 2
    width = max_point[1]-min_point[1] + 2 * edge_thickness
    high = max_point[0]-min_point[0] + 2 * edge_thickness
    zone = np.ones((high, width, 1))

    zone = img[(min_point[1]-edge_thickness):(min_point[1]+high), (min_point[0]-edge_thickness):(min_point[0]+width)]
    cv2.imshow('TABLE', zone)
    cv2.waitKey()


GetTable(image_rotate_cor, location, 2)

'''
def GetCell(img, location):
    # get ROI zone
    # location = [dot1, dot2, dot3, dot4, ...]
    ##################################
    # dot1----dot2-----dot3-----dot4 #
    #  |        |       |         |  #
    # dot5----dot6-----dot7-----dot8 #
    ##################################
    
    for dot in 
    zone = np.ones((200, 100, 1)) # define a 200*100 matrix, 1 mains 1 channel

    zone = img[200:400, 200:300] # write grayscale values into matrix, long from 200 to 400, hight from 200 to 300

    cv2.imshow("Zone", zone)
    cv2.waitKey(0)

    # fusion
    img[0:200, 0:100] = zone





pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


gray_image = cv2.imread(r'Development\imageTest\einfach_table.jpg', 0)
bina_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)

cv2.imshow('', bina_image)
cv2.waitKey()

result = pytesseract.image_to_string(bina_image)
print(type(result))
result.replace(' ', '\n')
re = result.split(' ')
print(re)
'''
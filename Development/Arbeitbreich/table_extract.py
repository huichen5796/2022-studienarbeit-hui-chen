'''
noch nicht fertig hier
- ROI of Zelle in table
- erode then dilate ---> reduce the noise
   
    ----------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    bina_image = ~cv2.erode(~bina_image, kernel,iterations = 1)   # erode to noise reduction
    # erode should after ROI before Tesseract
    ----------------
    
- Get Words in Zelle ---> Tesseract
- write Words in JISON
- JISON in Excal


'''

import pytesseract
import cv2
import numpy as np
import pandas as pd
from tilt_correction import TiltCorrection
from get_ROI_location import GetPoint


image_rotate_cor, white_image_cor = TiltCorrection(r'Development\imageTest\textandtable_0.png')
#image_rotate_cor, white_image_cor = TiltCorrection(r'Development\imageTest\einfach_table.jpg')
location = GetPoint(white_image_cor)[0]



def GetTable(img, location, edge_thickness):
    sum_row = list(np.sum(location, axis = 1)) # sum the x-axis and y-axis of point
    print(location)
    # print(sum_row)
    max_loca = sum_row.index(max(sum_row)) # The point in the lower right corner has the largest sum
    min_loca = sum_row.index(min(sum_row)) # The point in the lower right corner has the largest sum
    max_point = location[max_loca]
    min_point = location[min_loca]
    #print(max_point)
    #print(min_point)

    
    width = max_point[1]-min_point[1] + 2 * edge_thickness  # x2-x1+2e
    high = max_point[0]-min_point[0] + 2 * edge_thickness  # y2-y1+2e
    zone = np.ones((high, width, 1))

    zone = img[(min_point[0]-edge_thickness):(min_point[0]+high), (min_point[1]-edge_thickness):(min_point[1]+width)]
    cv2.imshow('TABLE', zone)
    cv2.waitKey()


GetTable(image_rotate_cor, location, 5)

'''
def GetCell(img, location):
    # get ROI zone
    # location = [dot1, dot2, dot3, dot4, ...]
    ##################################
    # dot1----dot2-----dot3-----dot4 #
    #  |        |       |         |  #
    # dot5----dot6-----dot7-----dot8 #
    ##################################                        [[ 17  19]
                                                               [ 17 375]
                                                               [ 17 641]
                                                               [ 17 907]
                                                               [ 65 375]
                                                               [ 65 641]
                                                               [ 65 907]
                                                               [ 66  19]  <------- bug !!!
                                                               # Disrupted the ordering and caused the region to not be closed
    
    for dot in 
    zone = np.ones(( , , 1)) 

    zone = img[] 

    cv2.imshow("Zone", zone)
    cv2.waitKey(0)
'''


def ExtrakText(image_cell):

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    result = pytesseract.image_to_string(image_cell)
    
    return result

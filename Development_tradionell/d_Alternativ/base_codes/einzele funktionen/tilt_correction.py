
### Neigungskorrektur durch Horizonalenerkennung
### bei rotating ist Scanverzerrung ignorriert.

'''
- main function ---> TiltCorrection(path)
  - in this function at first call the function LSDGetLines in get_linien to mark the lines
  - then call the function GetAngle()
  - then call the function ImageRotate()

  - input    ---  the path of the image we want to tilt correct
  - return1  ---  the tilt corrected image ---> is used for ROI and Tesseract
  - return2  ---  the tilt corrected white image ---> is uesd for get ROI location



- Schritte:
  - get the locations of lines
  - correct the picture by calculating the average deflection angle of all lines with deflection angles within +-45 degrees
''' 

import cv2
import numpy as np
import math
from get_linien import LSDGetLines



def GetAngle(lines):

    '''
    input     ---    list of the locations of lines, form [[x0,y0,x1,y1],[x2,y2,x3,y3],[...],[...],...] 
    output    ---    angle_average of horizonal lines    

    '''
   
    angle_list = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        if x1 == x2:
            angle_list.append(90)
        elif y1 == y2:
            angle_list.append(0)

        else:
            t = float(y2-y1)/(x2-x1)
            rotate_angle = math.degrees(math.atan(t))
            #angle_list.append(rotate_angle)
            #print(angle_list)

            '''
            # dabei können nur die gegen den Uhrzeigersinn geneigte Bilder korrigiert werden.
            if rotate_angle < 0:
                angle_list.append(rotate_angle)
            else:             
                rotate_angle1 = -(90 - rotate_angle)
                angle_list.append(rotate_angle1)     
            '''

            # dabei können nur die Bilder mit Neigungswinkeln innerhalb von +-45 Grad korrigiert werden.
            
            angle_list.append(rotate_angle)
            
    
    angle_list_45 = [angle_list[i] for i in range(len(angle_list)) if abs(angle_list[i])< 45] # list for angle < +-45
    #print(angle_list_45)
    #print(angle_list)

    if len(angle_list_45) == 0: # if no abs(anlge) < 45 
        angle_average = 0
    else: 
        angle_average = sum(angle_list_45)/len(angle_list_45)
 

    if __name__ == '__main__':
        print(angle_average)
        #print(angle_list)
    return angle_average
    


def ImageRotate(image, angle):

    '''
    input   ---  image: the image we want to rotate
    input   ---  angle: rotate anglel
    return  ---  the rotated image
    
    '''

    # get the shape of the image and then determine the rotate center

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
    image_rotate = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255,255,255))
    return image_rotate
    # borderValue default（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))
    


def TiltCorrection(path):

    '''
    input  --- the path of the image we want to tilt correct
    return --- the tilt corrected image

    '''
    
    gray_image = cv2.imread(path, 0)

    if __name__ == '__main__':
        cv2.imshow('original',gray_image)
        cv2.waitKey()
    
    lines, white_image = LSDGetLines(gray_image, 20)
    
    angle = GetAngle(lines)
    image_rotate_cor = ImageRotate(gray_image, angle)
    white_image_cor = ImageRotate(white_image, angle)
    

    if __name__ == '__main__':
        cv2.imshow('white_cor',white_image_cor)
        cv2.imshow('orig_cor', image_rotate_cor)
        cv2.waitKey()

    return image_rotate_cor, white_image_cor 
    
    
if __name__ == '__main__':
    TiltCorrection(r'Development\imageTest\textandtable_0.png')
    #TiltCorrection(r'Development\imageTest\einfach_table.jpg')
    #TiltCorrection(r'Development\imageTest\winkel_30.png')
    #TiltCorrection(r'Development\imageTest\winkel_-60.png')
    #TiltCorrection(r'Development\imageTest\winkel_60.png')
    
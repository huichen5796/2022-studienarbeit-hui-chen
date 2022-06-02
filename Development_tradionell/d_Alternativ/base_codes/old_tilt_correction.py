### Neigungskorrektur durch Horizonalenerkennung
'''
Schritte:
 1. LineSerch() --- markieren alle Horizonalen
 2. TiltCorrection() --- korrigieren Schiefe Bilder
''' 

import cv2
import numpy as np
import math
from binar_noise_reduction import GaussB

def LineSearch(bina_image):
    ### Line makieren durch HoughLines()
    edges = cv2.Canny(bina_image, 50, 250, apertureSize= 3) ## apertureSize is the size of kernel, also soble
    long_size = 400 # minlinelength

    lines = cv2.HoughLinesP(edges, 1.0, np.pi/180, 50, minLineLength=long_size, maxLineGap=5) 
    # https://blog.csdn.net/dcrmg/article/details/78880046 # 

    ### show the line
    # img_color = cv2.imread(path, -1)
    '''
    color_img = np.expand_dims(bina_img, axis=2)
    color_img = np.concatenate((color_img, color_img, color_img), axis=-1) 
    
    # change the gray image to color image
    '''
    ### oder: ###
    color_img = cv2.merge((bina_image, bina_image, bina_image))

    for line in lines:
        x1, y1, x2, y2 = line[0]

        img_line = cv2.line(color_img, (x1,y1), (x2, y2), (0, 0, 255), 1)
    
    if __name__ == '__main__':
        # cv2.imshow('input', bina_img)
        cv2.imshow("output", img_line)
        cv2.waitKey()

    return img_line, lines



def GetAngle(lines):
    angle_list = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        if x1 == x2 or y1 == y2:
            continue
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
            # dabei können nur die Bilder mit Neigungswinkeln innerhalb von 45 Grad korrigiert werden.
            if abs(rotate_angle) < 45:
                angle_list.append(rotate_angle)
            
    if len(angle_list) == 0:
        # print('Das Bild ist korrekt.')
        angle_average = 0
    else:
        angle_average = sum(angle_list)/len(angle_list)
        # bis hier bekommen wir die Neigungswinkel vom Bild
        # print('Der Neigungswinkel ist: ' + angle_average)

    


    if __name__ == '__main__':
        print(angle_average)
        # print(angle_list)
    return angle_average
    


def Rotate(image, angle):

    # get the shape of the image and then determine the center
    # https://blog.csdn.net/qq_44109682/article/details/117434461

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
    
    bina_image = GaussB(path)
    # bina_image = cv2.imread(path, 0)
    img_line, lines = LineSearch(bina_image)
    # b, gray_img, r = cv2.split(img_line)
    angle = GetAngle(lines)
    image_rotate_kor = Rotate(img_line, angle)
    # ret, image_rotate_kor = cv2.threshold(image_rotate_kor, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image_rotate_kor = cv2.cvtColor(image_rotate_kor,cv2.COLOR_BGR2GRAY)
    # information to cv2.cvtColor
    # RGB[A] --> Gray: Y <-- 0.299 R + 0.587 G + 0.114 B
    
    ret, image_rotate_kor = cv2.threshold(image_rotate_kor, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # gray_image to bina_image

    if __name__ == '__main__':
        cv2.imshow('korrigiertes Bild',image_rotate_kor)
        cv2.waitKey()

    return image_rotate_kor # is a bina_image
    
    
if __name__ == '__main__':
    TiltCorrection(r'Development\imageTest\rotate_table.png')
    #TiltCorrection(r'Development\imageTest\winkel_-30.png')
    #TiltCorrection(r'Development\imageTest\winkel_30.png')
    #TiltCorrection(r'Development\imageTest\winkel_-60.png')
    #TiltCorrection(r'Development\imageTest\winkel_60.png')
    


# bei rotating ist Scanverzerrung ignorriert.

'''
noch nicht fertig hier
- markieren linien durch HoughLinesP
- get location of table --- not done
# https://wenku.baidu.com/view/8bdffcf175a20029bd64783e0912a21614797fd3.html

'''
import cv2
import numpy as np
from get_linien import GetTable
import math
from distutils.command.config import LANG_EXT


def PointAndLineMark(bina_image):
    # https://blog.csdn.net/qq_33004317/article/details/100079230 #
    ### Line makieren durch HoughLines()
    edges = cv2.Canny(bina_image, 50, 250, apertureSize= 7) ## apertureSize is the size of kernel, also soble

    lines = cv2.HoughLinesP(edges, 1.0, np.pi/180, 50, minLineLength=20, maxLineGap=10) 

    # bis here get the locations of lines, but notice, 
    # If the line is too thick, a line will be detected as two lines that are very close together, 
    # which means that the line be [split], so they must be [merge] again.
    
    return lines


def DrawLine(bina_image, lines):
    ### show the line
    color_img = cv2.merge((bina_image, bina_image, bina_image))
    point_list = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        point_list.append((x1, y1))
        point_list.append((x2, y2))
        img_line = cv2.line(color_img, (x1,y1), (x2, y2), (0, 0, 255), 1)
    
    if __name__ == '__main__':
        cv2.imshow("output", img_line)
        cv2.waitKey()

    return img_line

def DrawPoint(img, point_list):

    for point in point_list:
        img_point = cv2.circle(img, point, 5, (0, 255, 0), 5)
    
    if __name__ == '__main__':
        cv2.imshow("output", img_point)
        cv2.waitKey()

    return img_point

def PointMerge(lines):
    # want to merge the nearby points in point_list by x value
    
    for i in range(len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        x11, y11, x22, y22 = lines[i+1][0]




def Main():
    bina_image = ~GetTable(r'Development\imageTest\einfach_table.jpg')  # einfache Tabelle
    # bina_image = GetTable(r'Development\imageTest\rotate_table.png')  # komplexe Tabelle
    PointAndLineMark(bina_image)
    
if __name__ == '__main__':
    Main()
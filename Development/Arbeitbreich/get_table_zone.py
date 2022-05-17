'''
- markieren linien durch HoughLinesP
- get location of table --- not done

'''
import cv2
import numpy as np
from get_linien import GetLine


def LineMark(bina_image):
    ### Line makieren durch HoughLines()
    edges = cv2.Canny(~bina_image, 50, 250, apertureSize= 3) ## apertureSize is the size of kernel, also soble

    lines = cv2.HoughLinesP(edges, 1.0, np.pi/180, 50, minLineLength=30, maxLineGap=10) 

    ### show the line
    color_img = cv2.merge((bina_image, bina_image, bina_image))

    for line in lines:
        x1, y1, x2, y2 = line[0]

        img_line = cv2.line(color_img, (x1,y1), (x2, y2), (0, 0, 255), 2)
    

    cv2.imshow("output", img_line)
    cv2.waitKey()

    return lines



def Main():
    bina_image = GetLine(r'Development\imageTest\rotate_table.png')
    LineMark(bina_image)
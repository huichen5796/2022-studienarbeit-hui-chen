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


def PointAndLineMark(bina_image):
    # https://blog.csdn.net/qq_33004317/article/details/100079230 #
    ### Line makieren durch HoughLines()
    edges = cv2.Canny(bina_image, 50, 250, apertureSize= 7) ## apertureSize is the size of kernel, also soble

    lines = cv2.HoughLinesP(edges, 1.0, np.pi/180, 50, minLineLength=20, maxLineGap=10) 

    # bis here get the locations of lines, but notice, 
    # If the line is too thick, a line will be detected as two lines that are very close together, 
    # which means that the line be [split], so they must be [merge] again.
    
    # merge the nearby line
    lines_merge = []
    for i in range(len(lines)-1): 
        # here must be len(lines)-1, cause the last line can not be merge with the next(the last has not the next)
        line1_x1, line1_y1, line1_x2, line1_y2 = lines[i]
        line2_x1, line2_y1, line2_x2, line2_y2 = lines[i+1]
        
        distance1 = math.sqrt((line2_x1-line1_x1)(line2_x1-line1_x1)+(line2_y1-line1_y1)(line2_y1-line1_y1))
        distance2 = math.sqrt((line2_x2-line1_x2)(line2_x2-line1_x2)+(line2_y2-line1_y2)(line2_y2-line1_y2))

        # 换个方法， 求一条直线的中点到另一条的距离
        
        if distance1 and distance2 < 3: # This value should be determined by the text size, temporarily set 3 for now
            x1_new = int((line1_x1 + line2_x1) / 2)
            y1_new = int((line1_y1 + line2_y1) / 2)
            x2_new = int((line1_x2 + line2_x2) / 2)
            y2_new = int((line1_y2 + line2_y2) / 2)
            lines_merge.append([x1_new, y1_new, x2_new, y2_new])
        
        else:
            lines_merge.append()






    ### show the line
    color_img = cv2.merge((bina_image, bina_image, bina_image))
    point_list = []
    for line in lines_merge:
        x1, y1, x2, y2 = line[0]
        
        point_list.append((x1, y1))
        point_list.append((x2, y2))
        img_line = cv2.line(color_img, (x1,y1), (x2, y2), (0, 0, 255), 2)
    for point in point_list:
        img_line_point = cv2.circle(img_line, point,5, (0, 255, 0), 5)
    

    cv2.imshow("output", img_line_point)
    cv2.waitKey()

    return lines_merge




def Main():
    bina_image = ~GetTable(r'Development\imageTest\einfach_table.jpg')  # einfache Tabelle
    # bina_image = GetTable(r'Development\imageTest\rotate_table.png')  # komplexe Tabelle
    PointAndLineMark(bina_image)
    
if __name__ == '__main__':
    Main()
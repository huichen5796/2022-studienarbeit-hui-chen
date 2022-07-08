import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import pytesseract
import time
from elasticsearch import Elasticsearch
es = Elasticsearch()


'''
**Main functions**

1. NoiseReducter(img)
    - cause now is the image with good quality, this function currently only has a binarization function (Gaussian-threshold)
    - input image
    - output binar image

2. TiltCorrection(image)
    - in this function at first call the function LSDGetLines to mark the lines
    - then call the function GetAngle() to get the average angle
    - then call the function ImageRotate() to correct the tilt

    - input    ---  the gray image we want to tilt correct
    - return1  ---  the tilt corrected image ---> is used for ROI and Tesseract
    - return2  ---  the tilt corrected white image ---> is uesd for get ROI location

    - Schritte:
    - get the locations of lines by copy image
    - correct the gray image by calculating the average deflection angle of all lines with deflection angles within +-45 degrees

5. GetPoint(copy_image)
    - in this function will at first call GetNode(), in GetNode() will call LineRow() to get horizonal lines by dilate und LineCol() to vertikal lines
      then call And_Border() to get image with big intersection point - Node
    - then call Concentrate() to concentrate the point to a point with only one pixel

    - input copy_rotate_cor
    - output1 --- location of intersection points
    - output2 --- image with only points

4. PointCorrection(location)
    # location = [dot1, dot2, dot3, dot4, ...]

    #   --------------> x-axis
    # : ##################################
    # : # dot1----dot2-----dot3-----dot4 #
    # ; #  |        |       |         |  #
    # y # dot5----dot6-----dot7-----dot8 #                         y   x
    #   ##################################                      [[ 16  18]
    #                                                            [ 16 374]
    #                                                            [ 16 640]
    #                                                            [ 16 906]
    #                                                            [ 64 374]
    #                                                            [ 64 640]
    #                                                            [ 64 906]
    #                                                            [ 65  18]  <------- bug !!!
    # Disrupted the ordering and caused the region to not be closed
    # Can't get correct cells with intersections that are not aligned, need to correction
    
    - Intersections with offsets within five pixels can be corrected.

5. GetTable(img, location, edge_thickness)
    - In fact, this is an extra step for the computer to extract the area where the table is located.
    - edge_thickness is the is the number of pixels we want to cut out, that is, the white space around the border.

6. ReadCell(location, image_rotate_cor, size)
    - read the info in the cell one by one from left to right from top to down
    - store the info in a list, list[0] is the info of the first row, ...

7. GetInfoDict(list_info)
    input
    - list_info

    return
    - dict_info // JSON ----> Each element in the dict is the content of the corresponding cell
                       the order is from left to right and from top to bottom

8. WriteData(dict_info)
    - write the dict_info in elasticsearch

9. Search(index)
    - search the data in elasticsearch
    

'''


def NoiseReducter(img):
    '''
     - cause now is the image with good quality, this function currently only has a binarization function
     - input path of color image
     - output binar image

    '''

    # Gauss Binar
    bina_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

    return bina_image


def LSDGetLines(img, long_size):
    '''
    - lines mark by LSD 
    - input is a gray image
    - output1 is the lines on it
    - output2 is a new white image with same shape of input image, on it is the lines of color image, location to location

    input - img             --- binar image
    input - long_size       --- min-long of the line
    return - dlines_long    --- dlines_long ---> a list for lines long than long_size in the image
    return - white_image    --- is a gray image

    '''

    copy_image = np.ones((img.shape[0], img.shape[1], 1))*255

    lsd = cv2.createLineSegmentDetector(0, scale=1)
    dlines = lsd.detect(img)

    dlines_long = []

    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))

        long = (y1-y0)*(y1-y0)+(x1-x0)*(x1-x0)
        if long >= long_size*long_size:
            cv2.line(copy_image, (x0, y0), (x1, y1), color=0,
                     thickness=3, lineType=cv2.LINE_AA)
            # the vorteil big thickness:
            # Draw a thick line to make two adjacent lines merge into one.
            # It can be seen here that there may be black dot noise in the center of the line when thickness small
            # which should be removed
            # But it will cause the intersection point to shift
            dlines_long.append(dline)

    if len(dlines_long) == 0:
        print('no table here')

    return dlines_long, copy_image


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
            # angle_list.append(rotate_angle)
            # print(angle_list)

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

    angle_list_45 = [angle_list[i] for i in range(len(angle_list)) if abs(
        angle_list[i]) < 45]  # list for angle < +-45
    # print(angle_list_45)
    # print(angle_list)

    if len(angle_list_45) == 0:  # if no abs(anlge) < 45
        angle_average = 0
    else:
        angle_average = sum(angle_list_45)/len(angle_list_45)

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
    image_rotate = cv2.warpAffine(
        image, M, (new_w, new_h), borderValue=(255, 255, 255))
    return image_rotate


def TiltCorrection(image):
    '''
    input  --- the image we want to tilt correct, must be gray image
    return --- the tilt corrected image

    '''

    lines, copy_image = LSDGetLines(image, 20)

    angle = GetAngle(lines)
    image_rotate_cor = ImageRotate(image, angle)
    copy_image_cor = ImageRotate(copy_image, angle)

    return image_rotate_cor, copy_image_cor


def LineRow(bina_image):  # get image only with row lines - get horizonal lines by dilate

    h, w = bina_image.shape
    hori_k = int(math.sqrt(w)*1.2)
    # hier für die Kernsize
    # https://blog.csdn.net/weixin_41189525/article/details/121889157
    kernel_hori = cv2.getStructuringElement(cv2.MORPH_RECT, (hori_k, 1))

    # white zone horizonal dilate then inversion
    image_row = ~cv2.dilate(~bina_image, kernel_hori, iterations=1)
    # white lines on black background now
    # Iterate twice, the first to restore the line length
    image_row = cv2.dilate(image_row, kernel_hori, iterations=3)
    # and the second to make the line longer
    # Make sure the resulting intersection shape is square
    return image_row


def LineCol(bina_image):  # get image only with col lines - get vertikal lines by dilate

    h, w = bina_image.shape

    vert_k = 20  # this parameter is difficult to find

    kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_k))
    image_col = cv2.dilate(~bina_image, kernel_vert, iterations=1)

    image_col = ~image_col
    image_col = cv2.dilate(image_col, kernel_vert, iterations=3)

    return image_col


def And_Border(img1, img2):  
    '''
    merge two images

    '''
    img1 = np.array(img1, np.uint8)
    img2 = np.array(img2, np.uint8)

    image_points = cv2.bitwise_and(img1, img2)

    return image_points


def GetNode(copy_image):
    '''
    call funtions LineRow(), LineCol(), And_Border()
    
    '''

    copy_image = copy_image.astype(np.uint8)

    ret, bina_image = cv2.threshold(~copy_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    image_row = LineRow(bina_image)
    image_col = LineCol(bina_image)

    #Border(image_row, image_col)
    image_points = And_Border(image_row, image_col)

    return image_points


def Concentrate(img):
    '''
    to concentrate the point to a point with only one pixel
    by calculating the coordinates of the midpoint of the square

    '''
    black_image = np.zeros((img.shape[0], img.shape[1]))
    contours, h = cv2.findContours(
        img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    #print(contours)
    for i in range(len(contours)):
        cnt = contours[i]
        x = int(np.average(cnt, axis=0)[0][0])
        y = int(np.average(cnt, axis=0)[0][1])
        #print(np.average(cnt, axis = 0))
        black_image[y, x] = 255

    location = np.argwhere(black_image > 127)

    # cv2.imshow('',black_image)
    # cv2.waitKey()
    return location, black_image


def GetPoint(copy_image):
    '''

    '''
    img = GetNode(copy_image)

    location, img_point = Concentrate(img)

    # cv2.imshow('',img_point)
    # cv2.waitKey()

    return location, img_point


def PointCorrection(location):
    # location = [dot1, dot2, dot3, dot4, ...]

    #   --------------> x-axis
    # : ##################################
    # : # dot1----dot2-----dot3-----dot4 #
    # ; #  |        |       |         |  #
    # y # dot5----dot6-----dot7-----dot8 #                         y   x
    #   ##################################                      [[ 16  18]
    #                                                            [ 16 374]
    #                                                            [ 16 640]
    #                                                            [ 16 906]
    #                                                            [ 64 374]
    #                                                            [ 64 640]
    #                                                            [ 64 906]
    #                                                            [ 65  18]  <------- bug !!!
    # Disrupted the ordering and caused the region to not be closed
    # Can't get correct cells with intersections that are not aligned, need to correction

    location = sorted(location, key=lambda x: x[0])

    for i in range(len(location)-1):
        if location[i+1][0] == location[i][0]:
            continue
        else:
            # suppose there are no cells with height less than 10
            if abs(location[i+1][0]-location[i][0]) < 10:
                location[i+1][0] = location[i][0]
            else:
                continue

    location = sorted(location, key=lambda x: (x[1], x[0]))

    for i in range(len(location)-1):
        if location[i+1][1] == location[i][1]:
            continue
        else:
            # suppose there are no cells with width less than 10
            if abs(location[i+1][1]-location[i][1]) < 10:
                location[i+1][1] = location[i][1]
            else:
                continue

    location = sorted(location, key=lambda x: (x[0], x[1]))

    return location


def GetTable(img, location, edge_thickness):
    '''
    # Method 1
    # sum the x-axis and y-axis of point
    sum_row = list(np.sum(location, axis=1))
    # The point in the lower right corner has the largest sum
    max_loca = sum_row.index(max(sum_row))
    # The point in the lower right corner has the largest sum
    min_loca = sum_row.index(min(sum_row))
    max_point = location[max_loca]
    min_point = location[min_loca]
    '''
    # Method 2
    max_point = location[-1]
    min_point = location[0]

    width = max_point[1]-min_point[1] + 2 * edge_thickness  # x2-x1+2e
    high = max_point[0]-min_point[0] + 2 * edge_thickness  # y2-y1+2e
    table_zone = np.ones((high, width, 1))

    table_zone = img[(min_point[0]-edge_thickness):(min_point[0]+high),
                     (min_point[1]-edge_thickness):(min_point[1]+width)]

    return table_zone


def Extrakt_Tesseract(image_cell):

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    result = pytesseract.image_to_string(image_cell)
    # print(result)

    if '\n' in result:
        result = result.replace('\n', '')
    return result


def ReadCell(location, image_rotate_cor, size):
    '''
    input:
     - location
     - img
     - size, is a parameter for: the size of the image to be cropped inward, thereby removing the border

    '''

    list_info = [[]]
    # get max of every col, for example: max_col[0] --> y max
    max_col = np.amax(location, axis=0)
    location_arr = np.array(location)[:, 0]
    #location_arr = location_arr.tolist()
    cell_number = 1
    for i in range(len(location)):
        x0 = location[i][1]
        # print(x0)
        y0 = location[i][0]
        if x0 < max_col[1] and y0 < max_col[0]:
            y1 = y0 + 1
            while y1 not in location_arr:
                y1 += 1

            x1 = location[i+1][1]

            width = x1-x0
            high = y1-y0

            cell_zone = np.ones((high-2*size, width-2*size, 1))
            cell_zone = image_rotate_cor[(y0+size):(y1-size), (x0+size):(x1-size)]
            #cell_zone = cv2.adaptiveThreshold(cell_zone, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
            cell_zone = NoiseReducter(cell_zone)

            result = Extrakt_Tesseract(cell_zone)

            list_info[-1].append(result)

            cell_number += 1

        elif x0 == max_col[1] and y0 < max_col[0]:
            list_info.append([])
        elif y0 == max_col[0]:
            del list_info[-1]
            break

    return list_info


def GetInfoDict(list_info):
    '''
    input
     - list_info

    return
     - dict_info // JSON ----> Each element in the dict is the content of the corresponding cell
                       the order is from left to right and from top to bottom

    '''

    key_list = []
    for i in range(len(list_info)):
        key_list.append('row%s' % (i+1))
    dict_info = dict(zip(key_list, list_info))

    if __name__ == '__main__':
        print(dict_info)

    return dict_info


def WriteData(dict_info):
    i = 1
    es.index(index='table', doc_type='_doc', body=dict_info)


def Search(index):
    """ 
    Searches for data in ES-index
    op: operator can be "and" (e.g. must match "Husten AND Fieber") or "or" (e.g. "Husten OR Fieber")
    """

    reqBody = {
        "size": 1000,  # no. of hits that will be sent
        "query": {
            "match_all": {}  # gives back all entries in ES-index
        }
    }

    res = es.search(index=index, body=reqBody)

    # preperation for pretty-print: encoding with utf-8 for "ä, ö, etc."
    data_print = json.dumps(res, indent=4, ensure_ascii=False).encode('utf8')
    # print(data_print.decode()) # pretty-print with indent level
    return data_print.decode()


def TableExtract(path):

    image = cv2.imread(path, 0)
    plt.subplot(1, 3, 1), plt.imshow(image, 'gray')
    plt.xticks([]), plt.yticks([])
    image_rotate_cor, copy_image_cor = TiltCorrection(image)
    plt.subplot(1, 3, 2), plt.imshow(copy_image_cor, cmap='gray')
    plt.xticks([]), plt.yticks([])
    location = GetPoint(copy_image_cor)[0]
    location = PointCorrection(location)
    table_zone = GetTable(image_rotate_cor, location, 2)
    plt.subplot(1, 3, 3), plt.imshow(table_zone, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()
    list_info = ReadCell(location, image_rotate_cor, 3)
    dict_info = GetInfoDict(list_info)
    WriteData(dict_info)


if __name__ == '__main__':
    es.indices.delete(index='table', ignore=[400, 404])  # deletes whole index

    TableExtract('Development_tradionell\\imageTest\\textandtablewinkel.png')
    time.sleep(1)
    print(Search('table'))

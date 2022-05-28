import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import json
from elasticsearch import Elasticsearch
es = Elasticsearch()


'''
1. NoiseReducter(path)
    - cause now is the image with good quality, this function currently only has a binarization function (Gaussian-threshold)
    - input path of image
    - output binar image

2. LSDGetLines(img, long_size)
    - it is the first two steps
    - color image to gray image
    - lines mark by LSD 

    - input is a gray image
    - output1 is the lines on it
    - output2 is a new white image with same shape of input image, on it is the lines of color image, location to location

3. TiltCorrection(bina_image)
    - in this function at first call the function LSDGetLines in get_linien to mark the lines
    - then call the function GetAngle()
    - then call the function ImageRotate()

    - input    ---  the bina image we want to tilt correct
    - return1  ---  the tilt corrected image ---> is used for ROI and Tesseract
    - return2  ---  the tilt corrected white image ---> is uesd for get ROI location

    - Schritte:
    - get the locations of lines
    - correct the picture by calculating the average deflection angle of all lines with deflection angles within +-45 degrees

4. GetPoint()
    - input white_rotate_cor ---> black lines white backgrund
    - output1 --- location of intersection points
    - output2 --- image with only points



'''

def NoiseReducter(path):

    '''
     - cause now is the image with good quality, this function currently only has a binarization function
     - input path of color image
     - output binar image

    '''
    gray_image = cv2.imread(path, 0) # 0 is the paraameter to get gray image

    # Gauss Binar   
    bina_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

    #if __name__ == '__main__':
    #    cv2.imshow('BinarGauss',bina_image)
    #    cv2.waitKey()

    return bina_image

def LSDGetLines(img, long_size):
    
    '''

    input - img             --- binar image
    input - long_size       --- min-long of the line
    return - dlines_long    --- dlines_long ---> a list for long lines in the image
    return - white_image    --- is a gray image
    
    '''
    
    white_image = np.ones((img.shape[0], img.shape[1], 1))
    
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
            cv2.line(white_image, (x0, y0), (x1, y1), color = 0, thickness = 3, lineType = cv2.LINE_AA)
            # the vorteil big thickness:
            # Draw a thick line to make two adjacent lines merge into one. 
            # It can be seen here that there may be black dot noise in the center of the line when thickness small
            # which should be removed
            # But it will cause the intersection point to shift
            dlines_long.append(dline)

    if len(dlines_long) == 0:
        print('no table here')

    return dlines_long, white_image

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

def TiltCorrection(bina_image):

    '''
    input  --- the path of the image we want to tilt correct
    return --- the tilt corrected image

    '''
    
    lines, white_image = LSDGetLines(bina_image, 20)
    
    angle = GetAngle(lines)
    image_rotate_cor = ImageRotate(bina_image, angle)
    white_image_cor = ImageRotate(white_image, angle)
    

    if __name__ == '__main__':
        cv2.imshow('white_cor',white_image_cor)
        cv2.imshow('orig_cor', image_rotate_cor)
        cv2.waitKey()

    return image_rotate_cor, white_image_cor 

def LineRow(bina_image):  # get image only with row lines - get horizonal lines

    
    h, w = bina_image.shape
    hori_k = int(math.sqrt(w)*1.2)
                                        # hier für die Kernsize 
                                        # https://blog.csdn.net/weixin_41189525/article/details/121889157
    kernel_hori = cv2.getStructuringElement(cv2.MORPH_RECT, (hori_k, 1))

    image_row = ~cv2.dilate(~bina_image, kernel_hori, iterations=1)  # white zone horizonal dilate then inversion
                                                                     # white lines on black background now
    image_row = cv2.dilate(image_row, kernel_hori, iterations=3)  # Iterate twice, the first to restore the line length 
                                                                  # and the second to make the line longer
                                                                  # Make sure the resulting intersection shape is square
                                                                    


    return image_row

def LineCol(bina_image):  # get image only with col lines - get vertikal lines
    
    h, w = bina_image.shape



    vert_k = 20  # this parameter is difficult to find

    kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_k))
    image_col = cv2.dilate(~bina_image, kernel_vert, iterations=1)
   

    image_col = ~image_col
    image_col = cv2.dilate(image_col, kernel_vert, iterations=3)
        
    return image_col

def And_Border(img1, img2): # useless function, just to show the table
    '''
    merge two images

    '''
    img1 = np.array(img1,np.uint8)
    img2 = np.array(img2,np.uint8)

    image_points = cv2.bitwise_and(img1, img2)

    #if __name__ == '__main__':
    #    cv2.imshow('Border', image_points)
    #    cv2.waitKey()

    return image_points

def GetNode(white_image):


    white_image = white_image.astype(np.uint8)


    ret, bina_image = cv2.threshold(~white_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        
    image_row = LineRow(bina_image)
    image_col = LineCol(bina_image)

    #Border(image_row, image_col)
    image_points = And_Border(image_row, image_col)

    return image_points
    
def Concentrate(img):
    black_image = np.zeros((img.shape[0], img.shape[1]))
    contours, h = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        x = int(np.average(cnt, axis = 0)[0][0])
        y = int(np.average(cnt, axis = 0)[0][1])
        #print(np.average(cnt, axis = 0))
        black_image[y,x] = 255

    location = np.argwhere(black_image > 127)
    
    #cv2.imshow('',black_image)
    #cv2.waitKey()
    return location, black_image

def GetPoint(white_image):
    
    '''

    '''
    img = GetNode(white_image)
    location, img_point = Concentrate(img)

    #print(location)
    #cv2.imshow('',img_point)
    #cv2.waitKey()
   
    
    return location, img_point

def GetTable(img, location, edge_thickness):
    sum_row = list(np.sum(location, axis = 1)) # sum the x-axis and y-axis of point
    # print(location)
    # print(sum_row)
    max_loca = sum_row.index(max(sum_row)) # The point in the lower right corner has the largest sum
    min_loca = sum_row.index(min(sum_row)) # The point in the lower right corner has the largest sum
    max_point = location[max_loca]
    min_point = location[min_loca]
    #print(max_point)
    #print(min_point)

    
    width = max_point[1]-min_point[1] + 2 * edge_thickness  # x2-x1+2e
    high = max_point[0]-min_point[0] + 2 * edge_thickness  # y2-y1+2e
    table_zone = np.ones((high, width, 1))

    table_zone = img[(min_point[0]-edge_thickness):(min_point[0]+high), (min_point[1]-edge_thickness):(min_point[1]+width)]

    #if __name__ == '__main__':
    #    cv2.imshow('TABLE', table_zone)
    #    cv2.waitKey()

    return table_zone


def PointCorrection(location):
    # get ROI zone
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
    #                                                            # Disrupted the ordering and caused the region to not be closed
    # need to correction

    location = sorted(location, key = lambda x:x[0])

    for i in range(len(location)-1):
        if location[i+1][0] == location[i][0]:
            continue
        else:
            if abs(location[i+1][0]-location[i][0]) < 10:  # suppose there are no cells with height less than 10
                location[i+1][0] = location[i][0]
            else:
                continue

    location = sorted(location, key = lambda x:(x[1], x[0]))

    for i in range(len(location)-1):
        if location[i+1][1] == location[i][1]:
            continue
        else:
            if abs(location[i+1][1]-location[i][1]) < 10:  # suppose there are no cells with width less than 10
                location[i+1][1] = location[i][1]
            else:
                continue

    location = sorted(location, key = lambda x:(x[0], x[1]))


    return location
'''
# test for PointCorrection
# [65,18] should be [64, 18]
# [64, 645] should be [64, 640]
location = [[ 16,  18],[ 16, 374],[ 16, 640],[ 16, 906],[ 64, 374],[ 64, 645],[ 64 ,906],[ 65,  18]]
location = PointCorrection(location)
print(location)
max = np.amax(location, axis=0)
print(max)
location = np.array(location)
max = location[:,0]
print(max)
'''
def GetCell(location, image_table, size, method):

    '''
    input:
     - img
     - location
     - size, is a parameter for: the size of the image to be cropped inward, thereby removing the border
     - method 
       - can be 1 ---> use Extrakt_Tesseract()
       - can be 2 ---> use Extrakt_Easyocr()
    
    '''
    
    list_info = [[]]
    max_col = np.amax(location, axis=0) # get max of every col, for example: max_col[0] --> y max
    location_arr = np.array(location)[:,0]
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
            cell_zone = image_table[(y0+size):(y1-size), (x0+size):(x1-size)]
            #cell_zone = cv2.adaptiveThreshold(cell_zone, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
            cell_zone = NoiseReducter(cell_zone)
            
            if method == 1:
                result = Extrakt_Tesseract(cell_zone)
            elif method == 2:
                result = Extrakt_Esayocr(cell_zone)
            list_info[-1].append(result)

            #if __name__ == '__main__':
            #    print(result)
            #    cv2.imshow('cell %s' %cell_number, cell_zone)
            #    cv2.waitKey()
                
            cell_number += 1
            
        elif x0 == max_col[1] and y0 < max_col[0]:
            list_info.append([])
        elif y0 == max_col[0]:
            del list_info[-1]
            break
    #if __name__ == '__main__':
    #    cv2.waitKey()


    return list_info



def GetInfoDict(path, thickness, size, method):
    '''
    input
     - path
     - thickness ----> should cut out more to get the full table
     - size ----> should be cut more inwards to remove cell borders

    return
     - dict_info // JSON ----> Each element in the dict is the content of the corresponding cell
                       the order is from left to right and from top to bottom
    
    '''
    image_rotate_cor, white_image_cor = TiltCorrection(path)
    #image_rotate_cor, white_image_cor = TiltCorrection(r'Development\imageTest\textandtablewinkel.png')
    location, img_point = GetPoint(white_image_cor)

    image_table = GetTable(image_rotate_cor, location, thickness)
    #image_table = cv2.adaptiveThreshold(image_table, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
    table_point = GetTable(img_point, location, thickness)
    # print(location)
    location = np.argwhere(table_point > 127)
    location = PointCorrection(location)
    # print(location)

    #cv2.imshow('TABLE', image_table)
    #cv2.imshow('TABLE_POINT', table_point)
    #cv2.waitKey()
    if __name__ == '__main__':
        plt.imshow(image_table, cmap = 'gray')
        plt.xticks([]),plt.yticks([])
        plt.show()

    list_info = GetCell(location, image_table, size, method)
    
    key_list = []
    for i in range(len(list_info)):
        key_list.append('row%s' %(i+1))
    dict_info = dict(zip(key_list, list_info))
    
    if __name__ =='__main__':
        print(dict_info)

    return dict_info

def WriteData(dict_info):
    i = 1
    es.index(index='table%s' %i, doc_type = '_doc', body = dict_info)


def Search(index):
    """ 
    Searches for data in ES-index
    op: operator can be "and" (e.g. must match "Husten AND Fieber") or "or" (e.g. "Husten OR Fieber")
    """

    reqBody = {
        "size": 1000, # no. of hits that will be sent
        "query": {
            "match_all": {} # gives back all entries in ES-index
        }
    }
    
    res = es.search(index=index, body=reqBody)

    data_print = json.dumps(res, indent=4, ensure_ascii=False).encode('utf8') # preperation for pretty-print: encoding with utf-8 for "ä, ö, etc."
    # print(data_print.decode()) # pretty-print with indent level
    return data_print.decode()


#dict_info = GetInfoDict(r'Development\imageTest\textandtablewinkel.png', 5, 3, 1)
#WriteData(dict_info)
#data_print = Search('table1')
#print(data_print)

es.indices.delete(index='table1', ignore=[400, 404]) # deletes whole index 

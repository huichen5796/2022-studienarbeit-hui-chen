'''
- ROI of Zelle in table
- erode then dilate ---> reduce the noise
- Get Words in Zelle ---> Tesseract

main function: ---> GetInfoList(path, thickness, size):
    
    input
     - path
     - thickness ----> should cut out more to get the full table
     - size ----> should be cut more inwards to remove cell borders

    return
     - list_info ----> Each element in the list is the content of the corresponding cell
                       the order is from left to right and from top to bottom
    
    

'''


import cv2
import numpy as np
from tilt_correction import TiltCorrection
from get_ROI_location import GetPoint
from binar_noise_reduction import NoiseReducter
from extrakt import Extrakt_Tesseract, Extrakt_Esayocr

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
def GetCell(location, image_table, size):

    '''
    input:
     - img
     - location
     - size, is a parameter for: the size of the image to be cropped inward, thereby removing the border
    
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

            result = Extrakt_Tesseract(cell_zone)
            #result = Extrakt_Esayocr(cell_zone)
            list_info[-1].append(result)

            if __name__ == '__main__':
                print(result)
                cv2.imshow('cell %s' %cell_number, cell_zone)
                cv2.waitKey()
                
            cell_number += 1
            
        elif x0 == max_col[1] and y0 < max_col[0]:
            list_info.append([])
        elif y0 == max_col[0]:
            del list_info[-1]
            break
    #if __name__ == '__main__':
    #    cv2.waitKey()


    return list_info



def GetInfoList(path, thickness, size):
    '''
    input
     - path
     - thickness ----> should cut out more to get the full table
     - size ----> should be cut more inwards to remove cell borders

    return
     - list_info ----> Each element in the list is the content of the corresponding cell
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

    cv2.imshow('TABLE', image_table)
    #cv2.imshow('TABLE_POINT', table_point)
    cv2.waitKey()

    list_info = GetCell(location, image_table, size)

    print(list_info)

    return list_info


if __name__ == '__main__':
    GetInfoList(r'Development\imageTest\textandtablewinkel.png', 5, 3)


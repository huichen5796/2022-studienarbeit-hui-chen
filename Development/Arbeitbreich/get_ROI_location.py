'''
- main function ---> GetPoint()
   - input white_rotate_cor ---> black lines white backgrund
   - output location of intersection point

'''
import cv2
import numpy as np
import math
from tilt_correction import TiltCorrection


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


def Or_Border(img1, img2): # useless function, just to show the table
    '''
    merge two images

    '''
    borders = cv2.bitwise_or(img1, img2)

    #if __name__ == '__main__':
    #    cv2.imshow('Border', borders)
    #    cv2.waitKey()

    return borders


def And_Border(img1, img2): # useless function, just to show the table
    '''
    merge two images

    '''
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
    
    # Go through each white point and turn the eight points around the point into black.
    # https://wenku.baidu.com/view/8bdffcf175a20029bd64783e0912a21614797fd3.html

    white = np.argwhere(img > 127) # img is a bina image so 0 is black, 255 is white
    
    # print(white)
    # print(white.shape[0])
    # print(white.shape)
    # print(white[1][1])
    
    for i in range(white.shape[0]): # Iterate over the coordinates of all white points
        c_row = white[i, 0] # get the number on i row 0 col ---> same as white[i][0] ----> x-axis
        c_col = white[i, 1] # ----> y-axis
        ###############
        ### 7 8 9 a ###
        ### 4 5 6 b ###   ---> now we got the location of number 7 (c_row, c_col)
        ### 1 2 3 c ###
        ### e r t y ###
        ############### 
        img[c_row, c_col+1] = 0      # change number 4 to black
        img[c_row+1, c_col] = 0      # number 8                  
        img[c_row+1, c_col+1] = 0    # number 5

        img[c_row+1, c_col+2] = 0    # 2
        img[c_row+2, c_col] = 0      # 9
        img[c_row, c_col+2] = 0      # 1
        img[c_row+2, c_col+1] = 0    # 6
        img[c_row+2, c_col+2] = 0    # 3
        
        # then got location of number 4, change 5 6 1 2 3 e r t to black
        # then 1
        # then e
        # then location of number 8
        # then 5
        # ... 
        # Eventually the top left pixel of each white square remains white and the rest becomes black.

        
    location = np.argwhere(img > 127)
    

    #if __name__ == '__main__':
    #    print(location)
    #    cv2.imshow('concentrate', img)
    #    cv2.waitKey()

    return location, img # is the location of white pixel


def GetPoint(white_image):

    '''
    input  --- the path of the image 

    '''
    img = GetNode(white_image)
    location, img_point = Concentrate(img)
    
    
    return location, img_point


if __name__ == '__main__':
    image_rotate_cor, white_image_cor = TiltCorrection(r'Development\imageTest\textandtable_0.png')
    
    location, img_point = GetPoint(white_image_cor)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    img_point = cv2.dilate(img_point, kernel, iterations=1)  

    merge_image = cv2.merge((And_Border(image_rotate_cor, ~img_point), And_Border(image_rotate_cor, ~img_point), image_rotate_cor))
    cv2.imshow('POINT_MARK', merge_image)
    cv2.waitKey()









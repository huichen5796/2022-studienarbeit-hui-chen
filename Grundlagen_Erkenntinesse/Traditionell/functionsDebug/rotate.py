import cv2
import numpy as np
import matplotlib.pyplot as plt

import math


# get the shape of the image and then determine the center
# https://blog.csdn.net/qq_44109682/article/details/117434461
def Rotate(path, angle):

    '''
    - path
    - angle:
        # 60 means rotate sixty degrees clockwise
        # -60 means rotate sixty degrees counterclockwise
    - return --- image_rotate

    '''
    angle = -angle

    image = cv2.imread(path, 0)


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
    image_rotate = cv2.warpAffine(image, M, (new_w, new_h),borderValue=(255,255,255))
    # borderValue default（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))


    cv2.imshow('',image_rotate)
    cv2.waitKey()
    cv2.imwrite('winkel_%s.png'%angle, image_rotate)

    return image_rotate

def GetAngle(img):
    '''
    - input: img, is gray

    - output: angle need to tilt correction

    '''
    bina_image = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)  # Gaussian binar
    bina_image1 = cv2.bitwise_not(bina_image)  # invert the image
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    # bina_image1 = cv2.erode(bina_image1,kernel,iterations = 1) # noise reduce by erode

    # get the location of white pixel
    coords = np.where(bina_image1 > 0)

    points = [None]*len(coords[0])
    for i,x in enumerate(coords[0]):
        y = coords[1][i]
        points[i] = (y,x)
    points = np.array(points)
    rect = cv2.minAreaRect(points)
    angle = int(rect[2])  # round all the white pixel by a rect

    # https://theailearner.com/tag/cv2-minarearect/
    # this function has three return
    # [0] -- center point of recttangle
    # [1] -- (w,h) of rectangle
    # [2] -- The rotation angle of the rectangle,
    # angle from the x-axis counterclockwise to w, the range is [-90,0)
    # actually after opencv4.5 is the angle in (0, 90]

    if __name__ == '__main__':
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.polylines(bina_image, [box], True, 0, 3)
        plt.subplot(121)
        plt.imshow(bina_image1, cmap = 'gray')
        plt.subplot(122)
        plt.imshow(bina_image, cmap = 'gray')
        plt.show()
    print(angle)
    if angle > 45:
        angle =  angle -90

    return angle


def ImageRotate(image, angle):
    '''
    - input 1: the image we want to rotate

    - input 2: rotate anglel, angle is positive for anti-clockwise and negative for clockwise.

    - output: the rotated image

    '''

    # get the shape of the image and then determine the rotate center

    h, w = image.shape[0:2]
    center_X, center_Y = w // 2, h // 2

    # make the rotation matrix
    M = cv2.getRotationMatrix2D((center_X, center_Y), angle, 1.0)
    # https://www.geeksforgeeks.org/python-opencv-getrotationmatrix2d-function/

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


def TiltCorrection(img):
    '''
    - input: the image we want to tilt correct, must be gray image
    - output: the tilt corrected image

    '''

    angle = GetAngle(img)

    image_rotate = ImageRotate(img, angle)
    plt.imshow(image_rotate)
    plt.show()
    return image_rotate

if __name__ == '__main__':
    for angle in range(0,91):
        #Rotate(r'Development\imageTest\test3_0.png', -angle)
        path = "winkel_%s.png" %angle
        img = cv2.imread(path,0)
        TiltCorrection(img)
    # 60 means rotate sixty degrees clockwise
    # -60 means rotate sixty degrees counterclockwise
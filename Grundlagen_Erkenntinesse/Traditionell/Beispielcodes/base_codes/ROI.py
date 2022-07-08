import cv2
import numpy as np


def GetROI(img):
    # get ROI zone

    '''
    maybe this function will be unsed for table extraction after determining the table position ###
    '''

    zone = np.ones((200, 100, 1)) # define a 200*100 matrix, 1 mains 1 channel

    zone = img[200:400, 200:300] # write grayscale values into matrix, long from 200 to 400, hight from 200 to 300

    cv2.imshow("Zone", zone)
    cv2.waitKey(0)

    # fusion
    img[0:200, 0:100] = zone

import cv2
import numpy as np
import matplotlib.pyplot as plt

def Binatization(path):

    # load the image
    img_gray = cv2.imread(path, 0)

    # img = cv2.imread(path)
    # gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # cv2.imshow("img_gray", img_gray)
    # cv2.waitKey()


    thresh1=cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    thresh2=cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    titles = ['img', 'BINARY', 'BINARY_INV']
    images = [img_gray,thresh1, thresh2]
    for i in range(3):
        plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
    
    return img_gray


# def ImageCompression(img):
    
    # Rauschunterdrückung



    

    


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


# def Binatization(img):
    #

Binatization('../Development/imageWithTable.png')

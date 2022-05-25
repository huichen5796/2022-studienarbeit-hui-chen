# https://www.csdn.net/tags/MtTaEg1sMDUxNDc4LWJsb2cO0O0O.html

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = r'Development\imageTest\rotate_table.png'

def LSD(img):
    image_gray = cv2.imread(img, 0)
    white = np.ones((image_gray.shape[0], image_gray.shape[1]))
    lsd = cv2.createLineSegmentDetector(0, scale=1)
    dlines = lsd.detect(image_gray)

    long_size = 20
    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))

        long = (y1-y0)*(y1-y0)+(x1-x0)*(x1-x0)
        if long >= long_size*long_size:
            cv2.line(white, (x0, y0), (x1, y1), 0, 1, cv2.LINE_AA)

    #cv2.imshow('', white)
    #cv2.waitKey()
    plt.imshow(white, cmap = 'gray')
    plt.xticks([]),plt.yticks([])
    plt.show()

def LineSearch(image):
    ### Line makieren durch HoughLines()
    image = cv2.imread(img, 0)
    bina_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
    edges = cv2.Canny(bina_image, 50, 250, apertureSize= 3) ## apertureSize is the size of kernel, also soble
    long_size = 400 # minlinelength

    lines = cv2.HoughLinesP(edges, 1.0, np.pi/180, 50, minLineLength=20, maxLineGap=5) 
    # https://blog.csdn.net/dcrmg/article/details/78880046 # 

    ### show the line
    # img_color = cv2.imread(path, -1)
    '''
    color_img = np.expand_dims(bina_img, axis=2)
    color_img = np.concatenate((color_img, color_img, color_img), axis=-1) 
    
    # change the gray image to color image
    '''
    ### oder: ###
    white = np.ones((image.shape[0], image.shape[1]))

    for line in lines:
        x1, y1, x2, y2 = line[0]

        img_line = cv2.line(white, (x1,y1), (x2, y2), 0, 1)
    
    if __name__ == '__main__':
        # cv2.imshow('input', bina_img)
        plt.imshow(white, cmap = 'gray')
        plt.xticks([]),plt.yticks([])
        plt.show()

    return img_line, lines

LineSearch(img)
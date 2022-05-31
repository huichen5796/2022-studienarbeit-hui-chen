
# Entfernen alle Linien und wandeln die Tabelle in eine rahmenlose Tabelle um

### Die Funktionalität ist hier vollständig, aber noch nicht als Funktion gekapselt. ###


import cv2
import matplotlib.pyplot as plt
import numpy as np

gray_image = cv2.imread('Development_tradionell\imageTest\einfach_table.jpg',0)
bina_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

def LSDGetLines(img):
    '''
    - lines mark by LSD 
    - input is a gray image
    - output is a new white image with same shape of input image, on it is the lines of image, location to location

    input - img             --- binar image
    return - white_image    --- is a bina image

    '''
    long_size = 20
    copy_image = np.zeros((img.shape[0], img.shape[1]))

    lsd = cv2.createLineSegmentDetector(0, scale=1)
    dlines = lsd.detect(img)

    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        long = (y1-y0)*(y1-y0)+(x1-x0)*(x1-x0)
        if long >= long_size*long_size:
            cv2.line(copy_image, (x0, y0), (x1, y1), color=255,
                        thickness=3, lineType=cv2.LINE_AA)
            
    return  copy_image



copy_image = LSDGetLines(gray_image)

def And_Border(img1, img2):  
    '''
    merge two images

    '''
    img1 = np.array(img1, np.uint8)
    img2 = np.array(img2, np.uint8)

    image = img1+img2

    return image

image = And_Border(copy_image, bina_image)
print(image.shape)
plt.imshow(image, cmap='gray')
plt.show()
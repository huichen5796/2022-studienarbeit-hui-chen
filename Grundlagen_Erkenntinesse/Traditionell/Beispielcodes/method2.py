
def LSDGetLines(img, minLong):
    '''
    lines be marked by FLD

    - input 1: is a bina image
    - input 2: the min long of lines

    - output 1: is a new black image with same shape of input image, on it is the lines of image, location to location
                be used for DeleteLine
    - output 2: list of lines, [[x0,y0,x1,y1],[x2,y2,x3,y3],[...],[...],...], for tiltCorrection

    '''

    # make a new black image with the same shape of input img
    copy_image = np.zeros((img.shape[0], img.shape[1]))

    lsd = cv2.createLineSegmentDetector(0, scale=1)
    # get all the location of lines by FLD, if no line, dlines = None
    dlines = lsd.detect(img)
    # print(dlines)
    longLines = []
    if dlines is not None:
        for dline in dlines[0]:
            x0 = int(round(dline[0][0]))
            y0 = int(round(dline[0][1]))
            x1 = int(round(dline[0][2]))
            y1 = int(round(dline[0][3]))
            long = (y1-y0)*(y1-y0)+(x1-x0)*(x1-x0)
            if long >= minLong*minLong:
                # It is possible that the shorter lines are part of the letter, so filter out the longer lines.
                cv2.line(copy_image, (x0, y0), (x1, y1), color=255,
                         thickness=10, lineType=cv2.LINE_AA)  # draw the white line on black image
                longLines.append([x0, y0, x1, y1])

    if __name__ == "__main__":

        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.savefig("47.svg", bbox_inches='tight',pad_inches = 0)
        plt.axis('off')
        plt.show()
        plt.imshow(copy_image, cmap="gray")
        plt.axis('off')
        plt.savefig("50.svg", bbox_inches='tight',pad_inches = 0)
        plt.axis('off')
        plt.show()

    return copy_image, longLines

def DeletLines(img):
    '''
    delet all the lines on image

    - input: gray image

    - output: bina image without lines

    '''
    img1 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 15)
    # _, img1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    img2 = LSDGetLines(img1, minLong=18)[0]

    img_deletline = OrImage(img1, img2)

    return img_deletline

def OrImage(img1, img2):
    '''
    add two images, weiss + weiss = weiss, weiss + schwarz = weiss, schwarz + schwarz = schwarz

    '''
    img1 = np.array(img1, np.uint8)
    img2 = np.array(img2, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img2 = cv2.dilate(img2, kernel, iterations=1)
    img2 = cv2.erode(img2, kernel, iterations=1)  # closing
    image = cv2.bitwise_or(img1, img2)
    # bitwise_or the original img and the copy img with black lines
    # so can the lines on image be deleted

    return image

'''
dilate the words, then find contours to get ROI then infos in the cell
'''

### Die Funktionalität ist hier vollständig, aber noch nicht als Funktion gekapselt. ###

import cv2
import matplotlib.pyplot as plt
import numpy as np

gray_image = cv2.imread('Development\\imageTest\\table41.png', 0)

plt.imshow(gray_image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.axis('off')
plt.savefig("100.svg", bbox_inches='tight',pad_inches = 0)
plt.show()

bina_image = DeletLines(gray_image)

plt.imshow(bina_image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.axis('off')
plt.savefig("108.svg", bbox_inches='tight',pad_inches = 0)
plt.show()

bina_image = cv2.bitwise_not(bina_image)
plt.imshow(bina_image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.axis('off')
plt.savefig("115.svg", bbox_inches='tight',pad_inches = 0)
plt.show()


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,9))
bina_image = cv2.dilate(bina_image,kernel,iterations = 1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,9))
bina_image = cv2.erode(bina_image,kernel,iterations = 1)
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
#bina_image = cv2.erode(bina_image,kernel,iterations = 1)

ret, bina_image = cv2.threshold(bina_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.imshow(bina_image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.axis('off')
plt.savefig("130.svg", bbox_inches='tight',pad_inches = 0)
plt.show()


contours, h = cv2.findContours(bina_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

list_contours = []
for cnt in contours:
    print(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    if w>10 and h>10:
        
        list_contours.append((x,y,w,h))
        cv2.rectangle(gray_image, (x,y), (x+w,y+h), 0, 2)

arr_contours = np.array(list_contours)

def HorizonalAlignment(location):
    

    location = sorted(location, key=lambda x: x[1])

    for i in range(len(location)-1):
        if location[i+1][1] == location[i][1]:
            continue
        else:
            # suppose there are no cells with height less than 10
            if abs(location[i+1][1]-location[i][1]) < 10:
                location[i+1][1] = location[i][1]
            else:
                continue


    location = sorted(location, key=lambda x: (x[1], x[0]))

    return location

arr_contours = HorizonalAlignment(arr_contours)

for cnt in arr_contours:
    x,y,w,h = cnt
    
    cv2.rectangle(gray_image, (x,y), (x+w,y+h), 0, 2)

plt.imshow(gray_image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()






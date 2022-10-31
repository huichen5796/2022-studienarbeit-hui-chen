import cv2
import matplotlib.pyplot as plt
import numpy as np

def FLDGetLines(img, minLong):
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

    fld = cv2.ximgproc.createFastLineDetector()
    # get all the location of lines by FLD, if no line, dlines = None
    dlines = fld.detect(img)
    # print(dlines)
    longLines = []
    if dlines is not None:
        for dline in dlines:
            x0 = int(round(dline[0][0]))
            y0 = int(round(dline[0][1]))
            x1 = int(round(dline[0][2]))
            y1 = int(round(dline[0][3]))
            long = (y1-y0)*(y1-y0)+(x1-x0)*(x1-x0)
            if long >= minLong*minLong:
                # It is possible that the shorter lines are part of the letter, so filter out the longer lines.
                cv2.line(copy_image, (x0, y0), (x1, y1), color=255,
                         thickness=3, lineType=cv2.LINE_AA)  # draw the white line on black image
                longLines.append([x0, y0, x1, y1])

    return copy_image, longLines

def OrImage(img1, img2):
    '''
    add two images, weiss + weiss = weiss, weiss + schwarz = weiss, schwarz + schwarz = schwarz

    '''
    img1 = np.array(img1, np.uint8)
    img2 = np.array(img2, np.uint8)

    plt.subplot(111)
    plt.imshow(img1,'gray')
    plt.axis('off')
    plt.savefig('1.svg', dpi=300, bbox_inches="tight")

    plt.subplot(111)
    plt.imshow(img2,'gray')
    plt.axis('off')
    plt.savefig('2.svg', dpi=300, bbox_inches="tight")

    image = cv2.bitwise_or(img1, img2)

    plt.subplot(111)
    plt.imshow(image,'gray')
    plt.axis('off')
    plt.savefig('3.svg', dpi=300, bbox_inches="tight")

    return image

def DeletLines(img):
    '''
    delet all the lines on image

    - input: gray image

    - output: bina image without lines

    '''
    img1 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 10)

    img2 = FLDGetLines(img1, minLong=40)[0]

    img_deletline = OrImage(img1, img2)

    return img_deletline

gray_image = cv2.imread('zbild\\table\\table11.png', 0)

bina_image = DeletLines(gray_image)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,2))

bina_image = cv2.bitwise_not(bina_image)

bina_image = cv2.dilate(bina_image, kernel, iterations = 2)

bina_image = cv2.erode(bina_image, kernel, iterations = 3)

bina_image = cv2.dilate(bina_image, kernel, iterations = 1)

plt.subplot(111)
plt.imshow(bina_image,'gray')
plt.axis('off')
plt.savefig('4.svg', dpi=300, bbox_inches="tight")



'''
Bei einer Tabelle ohne vertikale Linien oder mit wenigen vertikalen Linien kann bestimmt werden, 
ob eine Pixelspalte zu Text oder zu einem Zwischenraum zwischen Texten gehört, 
indem die Summe der Pixel pro Spalte berechnet wird. Auf die gleiche Weise summieren und reihenweise beurteilen.
Fügen dann entsprechende Zeilen hinzu.

'''

import numpy as np
import matplotlib.pyplot as plt
import cv2


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
        plt.savefig("47.svg", bbox_inches='tight', pad_inches=0)
        plt.axis('off')
        plt.show()
        plt.imshow(copy_image, cmap="gray")
        plt.axis('off')
        plt.savefig("50.svg", bbox_inches='tight', pad_inches=0)
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


# table line fix
### Die Funktionalität ist hier vollständig, aber noch nicht als Funktion gekapselt. ###


gray_image = cv2.imread('Development\\imageTest\\filter.png', 0)
gray_image = cv2.resize(
    gray_image, (gray_image.shape[1]//2, gray_image.shape[0]//2))

plt.imshow(gray_image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.axis('off')
plt.savefig("100.svg", bbox_inches='tight', pad_inches=0)
plt.show()


bina_image = cv2.adaptiveThreshold(
    gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

# bina_image = DeletLines(bina_image)
# plt.imshow(bina_image, cmap='gray')
# plt.axis('off')
# plt.savefig("107.svg", bbox_inches='tight',pad_inches = 0)
# plt.show()


def ZeroOrOne(x):  # Hier ist eine einfache Binärisierungsfunktion
    x[x < 127] = 0  # schwarz
    x[x > 127] = 1  # weiss

    # so the spalt or row without text has the biggest sum

    return x


bina_image = ZeroOrOne(bina_image)


sum_col = bina_image.sum(axis=0)
sum_row = bina_image.sum(axis=1)
print(sum_col)
print(sum_row)
print(bina_image.shape)
h, w = bina_image.shape


plt.plot(range(w), sum_col)
plt.axis('off')
plt.savefig("129.svg", bbox_inches='tight', pad_inches=0)
plt.show()


plt.plot(sum_row, range(h))
plt.axis('off')
plt.savefig("134.svg", bbox_inches='tight', pad_inches=0)
plt.show()

ax = plt.gca()
ax.xaxis.set_ticks_position('top')
ax.invert_yaxis()


max_col = max(sum_col)
en = enumerate(sum_col)
list_maxcol = [i for i, n in en if n == max_col]
# print(list_maxcol)

copy_image1 = np.zeros((h, w))
for col in list_maxcol:
    copy_image1[:, col] = 1

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
copy_image1 = cv2.erode(copy_image1, kernel, iterations=1)
copy_image1 = cv2.dilate(copy_image1, kernel, iterations=1)
copy_image1 = np.array(copy_image1, np.uint8)
contours, hen = cv2.findContours(
    copy_image1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    cnt = contours[i]
    x = int(np.average(cnt, axis=0)[0][0])

    bina_image[:, x] = 0


plt.imshow(copy_image1, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.axis('off')
plt.savefig("164.svg", bbox_inches='tight', pad_inches=0)
plt.show()


max_row = max(sum_row)
en = enumerate(sum_row)
list_maxrow = [i for i, n in en if n == max_row]
# print(list_maxrow)

copy_image2 = np.zeros((h, w))
for row in list_maxrow:
    copy_image2[row, :] = 1


copy_image2 = cv2.erode(copy_image2, kernel, iterations=1)
copy_image2 = cv2.dilate(copy_image2, kernel, iterations=1)
copy_image2 = np.array(copy_image2, np.uint8)
contours, hen = cv2.findContours(
    copy_image2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    cnt = contours[i]
    y = int(np.average(cnt, axis=0)[0][1])

    bina_image[y, :] = 0

#image = cv2.bitwise_or(copy_image1, copy_image2)


plt.imshow(copy_image2, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.axis('off')
plt.savefig("193.svg", bbox_inches='tight', pad_inches=0)
plt.show()


plt.imshow(bina_image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.axis('off')
plt.savefig("201.svg", bbox_inches='tight', pad_inches=0)
plt.show()

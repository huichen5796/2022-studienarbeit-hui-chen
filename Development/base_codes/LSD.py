# https://www.csdn.net/tags/MtTaEg1sMDUxNDc4LWJsb2cO0O0O.html

import cv2
import numpy as np

img = r'Development\imageTest\rotate_table.png'
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

cv2.imshow('', white)
cv2.waitKey()
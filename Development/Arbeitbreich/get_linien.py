
'''

- it is the first two steps
  - color image to gray image
  - lines mark by LSD 

- input is a gray image
- output1 is the lines on it
- output2 is a new white image, on it is the lines of color image, location to location

'''

import cv2
import numpy as np


def LSDGetLines(img, long_size):
    '''
    input - img             --- gray image
    input - long_size       --- min-long of the line
    return - dlines_long    --- dlines_long ---> a list for long lines in the image
    return - white_image    --- is a gray image
    
    '''
    
    white_image = np.ones((img.shape[0], img.shape[1], 1))*255
    
    lsd = cv2.createLineSegmentDetector(0, scale=1)
    dlines = lsd.detect(img)

    dlines_long = []
    
    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))

        long = (y1-y0)*(y1-y0)+(x1-x0)*(x1-x0)
        if long >= long_size*long_size:
            cv2.line(white_image, (x0, y0), (x1, y1), color = 0, thickness = 2, lineType = cv2.LINE_AA)
            # the reason of big thickness:
            # Draw a thick line to make two adjacent lines merge into one. 
            # It can be seen here that there may be black dot noise in the center of the line when thickness small
            # which should be removed
            dlines_long.append(dline)
    
    if __name__ == '__main__':
        cv2.imshow('', white_image)
        cv2.waitKey()

    if len(dlines_long) == 0:
        print('no table here')

    return dlines_long, white_image

if __name__ == '__main__':
    # img = GaussB(r'Development\imageTest\rotate_table.png')
    img = cv2.imread(r'Development\imageTest\textandtable_0.png', 0)
    cv2.imshow('', img)
    cv2.waitKey()
    # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    LSDGetLines(img, 20)
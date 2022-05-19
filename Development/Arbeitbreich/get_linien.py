
'''
- tilt correction
- use LSD get Linien

'''

import cv2
from tilt_correction import TiltCorrection
import numpy as np

def LSDGetLines(img, long_size):
    '''
    img        --- the image we want to get line
    long_size  --- min-long of the line
    return     --- dlines_long ---> a list for long lines in the image
    
    '''
    
    white_image = np.ones((img.shape[0], img.shape[1]))
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
            cv2.line(white_image, (x0, y0), (x1, y1), 0, 1, cv2.LINE_AA)
            dlines_long.append(dline)
    
    if __name__ == '__main__':
        cv2.imshow('', white_image)
        cv2.waitKey()

    return dlines_long

if __name__ == '__main__':
    img = TiltCorrection(r'Development\imageTest\rotate_table.png')
    LSDGetLines(img, 20)
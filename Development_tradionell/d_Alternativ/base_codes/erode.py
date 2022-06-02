import numpy as np

# https://zhuanlan.zhihu.com/p/411909082
# https://blog.csdn.net/weixin_42218981/article/details/116182983
# https://www.cnblogs.com/wuyuan2011woaini/p/15656053.html


import cv2
import numpy as np
 
 
img = cv2.imread('Development\imageTest\einfach_table.jpg',0)
img = ~cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
erode = cv2.erode(img,kernel,iterations = 1)

cv2.imshow("org",img)
cv2.imshow("result", erode)

cv2.waitKey(0)
cv2.destroyAllWindows()





def horizon_erode(bin_img):
    out_img = np.zeros(shape=bin_img.shape, dtype=np.uint8) + 255
    h = bin_img.shape[0]
    w = bin_img.shape[1]
    for i in range(h):
        for j in range(1,w-1):
            out_img[i][j]=0
            for k in range(3):
                if bin_img[i][j+k-1] > 127:
                    out_img[i][j]=255
    return out_img


def vertical_erode(bin_img):
    out_img = np.zeros(shape=bin_img.shape, dtype=np.uint8) + 255
    h = bin_img.shape[0]
    w = bin_img.shape[1]
    for i in range(1,h-1):
        for j in range(w):
            out_img[i][j]=0
            for k in range(3):
                if bin_img[i+k-1][j] > 127:
                    out_img[i][j]=255
    return out_img


def all_erode(bin_img):
    out_img = np.zeros(shape=bin_img.shape, dtype=np.uint8) + 255
    h = bin_img.shape[0]
    w = bin_img.shape[1]
    B=[1,0,1,0,0,0,1,0,1]
    for i in range(1,h-1):
        for j in range(1,w-1):
            out_img[i][j]=0
            for m in range(3):
                for n in range(3):
                    if B[m*3+n] == 1:
                        continue
                    if bin_img[i+m-1][j+n-1] > 127:
                        out_img[i][j]=255
    return out_img
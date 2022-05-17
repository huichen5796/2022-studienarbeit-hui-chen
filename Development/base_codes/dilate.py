import cv2
import numpy as np
 
 
img = cv2.imread('Development\imageTest\einfach_table.jpg',0)
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,2))
dict = cv2.dilate(img,kernel,iterations = 1)
cv2.imshow("org",img)
cv2.imshow("result", dict)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Quelle: https://blog.csdn.net/weixin_43624538/article/details/87436154
import cv2
import numpy as np

img = cv2.imread(r'Development\imageTest\rotate_table.png')

b, g, r = cv2.split(img)
merged = cv2.merge([b,g,r])
zeros = np.zeros(img.shape[:2], dtype="uint8")
merged_r = cv2.merge([zeros, zeros, r])

cv2.imshow('imgage',img)
cv2.imshow('Blue', b)
cv2.imshow('Green', g)
cv2.imshow('Red', r)
cv2.imshow('Merged', merged)
cv2.imshow('merge_r', merged_r)
cv2.waitKey()
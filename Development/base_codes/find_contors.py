# https://blog.csdn.net/weixin_43869605/article/details/119921444?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-119921444-blog-99675967.pc_relevant_scanpaymentv1&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-119921444-blog-99675967.pc_relevant_scanpaymentv1&utm_relevant_index=1

# https://wenku.baidu.com/view/a5ccf309de36a32d7375a417866fb84ae45cc3ca.html
# https://www.freesion.com/article/9470170204/
import cv2
from cv2 import waitKey
import numpy as np

'''
img = cv2.imread(r'Development\imageTest\one.png',0)
c, h = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
print(c)
x,y,w,h = cv2.boundingRect(img)
img = cv2.merge((img,img,img))
cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
cv2.imshow('',img)
cv2.waitKey()

img = cv2.imread(r'Development\imageTest\conc.png',0)
c, h = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
img = cv2.approxPolyDP(c[0],10,True)
print(img)
x,y,w,h = cv2.boundingRect(img)
#img = cv2.merge((img,img,img))
#cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
#cv2.imshow('',img)
#cv2.waitKey()
print(img)
'''
print(np.zeros((1,1,1)))
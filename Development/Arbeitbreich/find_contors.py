# https://blog.csdn.net/weixin_43869605/article/details/119921444?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-119921444-blog-99675967.pc_relevant_scanpaymentv1&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-119921444-blog-99675967.pc_relevant_scanpaymentv1&utm_relevant_index=1

# https://wenku.baidu.com/view/a5ccf309de36a32d7375a417866fb84ae45cc3ca.html
# https://www.freesion.com/article/9470170204/
import cv2
from tilt_correction import TiltCorrection
import numpy as np
img1, img2 = TiltCorrection(r'Development\imageTest\einfach_table.jpg')
ret,img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
img2 = img2.astype(np.uint8)
contours,h = cv2.findContours(img2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
n=len(contours)       #轮廓个数
contoursImg=[]
for i in range(n):
    length = cv2.arcLength(contours[i], True)  #获取轮廓长度
    area = cv2.contourArea(contours[i])        #获取轮廓面积
    print('length['+str(i)+']长度=',length)
    print("contours["+str(i)+"]面积=",area)
    temp=np.zeros(img2.shape,np.uint8) #生成黑背景
    contoursImg.append(temp)
    contoursImg[i]=cv2.drawContours(contoursImg[i],contours,i,(255,255,255), 3)  #绘制轮廓
    cv2.imshow("contours[" + str(i)+"]",contoursImg[i])   #显示轮廓
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow('',img2)
cv2.waitKey()
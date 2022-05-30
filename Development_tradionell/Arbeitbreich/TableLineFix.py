import cv2
import matplotlib.pyplot as plt
import numpy as np

gray_image = cv2.imread('Development_tradionell\\imageTest\\tabelle_ohne_linien.png', 0)

plt.subplot(2, 2, 3)
plt.imshow(gray_image, cmap='gray')
plt.xticks([]), plt.yticks([])


bina_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)

def ZeroOrOne(x):
    x[x<127] = 0
    x[x>127] = 1
    return x
bina_image = ZeroOrOne(bina_image)

sum_col = bina_image.sum(axis=0)
sum_row = bina_image.sum(axis=1)
print(sum_col)
print(sum_row)
print(bina_image.shape)
h,w = bina_image.shape


plt.subplot(2,2,1)
plt.plot(range(w),sum_col)



plt.subplot(2,2,4)
plt.plot(sum_row, range(h))

ax = plt.gca()                               
ax.xaxis.set_ticks_position('top')
ax.invert_yaxis()  

plt.show()

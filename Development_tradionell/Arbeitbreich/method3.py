
'''
Bei einer Tabelle ohne vertikale Linien oder mit wenigen vertikalen Linien kann bestimmt werden, 
ob eine Pixelspalte zu Text oder zu einem Zwischenraum zwischen Texten gehört, 
indem die Summe der Pixel pro Spalte berechnet wird. Auf die gleiche Weise summieren und reihenweise beurteilen.
Fügen dann entsprechende Zeilen hinzu.

'''



### table line fix
### Die Funktionalität ist hier vollständig, aber noch nicht als Funktion gekapselt. ###
 

import cv2
import matplotlib.pyplot as plt
import numpy as np

gray_image = cv2.imread('Development_tradionell\\imageTest\\tabelle_ohne_linien.png', 0)

plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.xticks([]), plt.yticks([])


bina_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

def ZeroOrOne(x): ## Hier ist eine einfache Binärisierungsfunktion
    x[x<127] = 0 # schwarz
    x[x>127] = 1 # weiss

    # so the spalt or row without text has the biggest sum

    return x
bina_image = ZeroOrOne(bina_image)


sum_col = bina_image.sum(axis=0)
sum_row = bina_image.sum(axis=1)
print(sum_col)
print(sum_row)
print(bina_image.shape)
h,w = bina_image.shape


#plt.subplot(2,2,1)
#plt.plot(range(w),sum_col)

#plt.subplot(2,2,4)
#plt.plot(sum_row, range(h))

#ax = plt.gca()                               
#ax.xaxis.set_ticks_position('top')
#ax.invert_yaxis()  


max_col = max(sum_col)
en = enumerate(sum_col)
list_maxcol = [i for i, n in en if n == max_col]
#print(list_maxcol)

copy_image1 = np.zeros((h, w))
for col in list_maxcol:
    copy_image1[:,col] = 1

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
copy_image1 = cv2.erode(copy_image1,kernel,iterations = 1)
copy_image1 = cv2.dilate(copy_image1,kernel,iterations = 1)
copy_image1 = np.array(copy_image1,np.uint8)
contours, hen = cv2.findContours(copy_image1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    cnt = contours[i]
    x = int(np.average(cnt, axis=0)[0][0])

    gray_image[:, x] = 0


plt.subplot(2,2,3)
plt.imshow(copy_image1, cmap='gray')
plt.xticks([]), plt.yticks([])


max_row = max(sum_row)
en = enumerate(sum_row)
list_maxrow = [i for i, n in en if n == max_row]
#print(list_maxrow)

copy_image2 = np.zeros((h, w))
for row in list_maxrow:
    copy_image2[row,:] = 1



copy_image2 = cv2.erode(copy_image2,kernel,iterations = 1)
copy_image2 = cv2.dilate(copy_image2,kernel,iterations = 1)
copy_image2 = np.array(copy_image2,np.uint8)
contours, hen = cv2.findContours(copy_image2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    cnt = contours[i]
    y = int(np.average(cnt, axis=0)[0][1])

    gray_image[y, :] = 0

#image = cv2.bitwise_or(copy_image1, copy_image2)

plt.subplot(2,2,2)
plt.imshow(copy_image2, cmap='gray')
plt.xticks([]), plt.yticks([])




plt.subplot(2,2,4)
plt.imshow(gray_image, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.show()
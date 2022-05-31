'''
### table line fix
### nicht effizient, Ich dachte, es wäre von Lärm betroffen, also habe ich es aufgegeben # 

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

plt.subplot(2, 2, 2)
plt.imshow(bina_image, cmap='gray')
plt.xticks([]), plt.yticks([])

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


'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
from TableExtract import NoiseReducter, Extrakt_Tesseract

gray_image = cv2.imread('Development_tradionell\imageTest\\tabelle_ohne_linien.png', 0)
bina_image = ~cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
bina_image = cv2.dilate(bina_image,kernel,iterations = 1)
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
#bina_image = cv2.erode(bina_image,kernel,iterations = 1)

plt.imshow(bina_image, cmap='gray')
plt.show()

img = bina_image
contours, h = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

list_contours = []
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    if w>20 and h>20:
        
        list_contours.append((x,y,w,h))
        cv2.rectangle(gray_image, (x,y), (x+w,y+h), 0, 2)
plt.imshow(gray_image, cmap='gray')
plt.show()

arr_contours = np.array(list_contours)

def HorizonalAlignment(location):
    

    location = sorted(location, key=lambda x: x[1])

    for i in range(len(location)-1):
        if location[i+1][1] == location[i][1]:
            continue
        else:
            # suppose there are no cells with height less than 10
            if abs(location[i+1][1]-location[i][1]) < 10:
                location[i+1][1] = location[i][1]
            else:
                continue


    location = sorted(location, key=lambda x: (x[1], x[0]))

    return location

arr_contours = HorizonalAlignment(arr_contours)
print(arr_contours)

'''
def ReadCell(location, image_rotate_cor, size):
    

    list_info = [[]]
    # get max of every col, for example: max_col[0] --> y max
    max_col = np.amax(location, axis=0)
    location_arr = np.array(location)[:, 0]
    #location_arr = location_arr.tolist()
    cell_number = 1
    for i in range(len(location)):
        x0 = location[i][0]
        # print(x0)
        y0 = location[i][1]
        if x0 < max_col[0] and y0 < max_col[1]:
            y1 = y0 + 1
            while y1 not in location_arr:
                y1 += 1

            x1 = location[i+1][0]

            width = x1-x0
            high = y1-y0

            cell_zone = np.ones((high-2*size, width-2*size, 1))
            cell_zone = image_rotate_cor[(y0+size):(y1-size), (x0+size):(x1-size)]
            #cell_zone = cv2.adaptiveThreshold(cell_zone, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
            cell_zone = NoiseReducter(cell_zone)

            result = Extrakt_Tesseract(cell_zone)

            list_info[-1].append(result)

            cell_number += 1

        elif x0 == max_col[1] and y0 < max_col[0]:
            list_info.append([])
        elif y0 == max_col[0]:
            del list_info[-1]
            break

    return list_info



'''





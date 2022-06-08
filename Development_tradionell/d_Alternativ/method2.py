

'''
dilate the words, then find contours to get ROI then infos in the cell
'''

### Die Funktionalität ist hier vollständig, aber noch nicht als Funktion gekapselt. ###

import cv2
import matplotlib.pyplot as plt
import numpy as np
from TableExtract import NoiseReducter, Extrakt_Tesseract, DeletLines, TiltCorrection

gray_image = cv2.imread('Development_tradionell\\imageTest\\rotate_table.png', 0)
gray_image = TiltCorrection(gray_image)

bina_image = DeletLines(gray_image)

plt.imshow(bina_image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()
bina_image = cv2.bitwise_not(bina_image)
plt.imshow(bina_image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
bina_image = cv2.erode(bina_image,kernel,iterations = 1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
bina_image = cv2.dilate(bina_image,kernel,iterations = 1)
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
#bina_image = cv2.erode(bina_image,kernel,iterations = 1)

ret, bina_image = cv2.threshold(bina_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.imshow(bina_image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()


contours, h = cv2.findContours(bina_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

list_contours = []
for cnt in contours:
    print(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    if w>10 and h>10:
        
        list_contours.append((x,y,w,h))
        cv2.rectangle(gray_image, (x,y), (x+w,y+h), 0, 2)

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

for cnt in arr_contours:
    x,y,w,h = cnt
    
    cv2.rectangle(gray_image, (x,y), (x+w,y+h), 0, 2)

plt.imshow(gray_image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()



def ReadCell(location, image_rotate_cor):
    
    list_info = [[]]
    x1,y1,w1,h1 = location[0]
    cell_zone = np.ones((h1, w1, 1))
    cell_zone = image_rotate_cor[(y1):(y1+h1), (x1):(x1+w1)]
    #cell_zone = cv2.adaptiveThreshold(cell_zone, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
    cell_zone = NoiseReducter(cell_zone)

    result = Extrakt_Tesseract(cell_zone)

    list_info[-1].append(result)

    cell_number = 2
    for i in range(1, len(location)):
        if location[i][1] == location[i-1][1]:
            
            x,y,w,h = location[i]

            cell_zone = np.ones((h, w, 1))
            cell_zone = image_rotate_cor[(y):(y+h), (x):(x+w)]
            #cell_zone = cv2.adaptiveThreshold(cell_zone, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
            cell_zone = NoiseReducter(cell_zone)

            result = Extrakt_Tesseract(cell_zone)

            list_info[-1].append(result)

            cell_number += 1

        else:
            list_info.append([])
            x,y,w,h = location[i]

            cell_zone = np.ones((h, w, 1))
            cell_zone = image_rotate_cor[(y):(y+h), (x):(x+w)]
            #cell_zone = cv2.adaptiveThreshold(cell_zone, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
            cell_zone = NoiseReducter(cell_zone)

            result = Extrakt_Tesseract(cell_zone)

            list_info[-1].append(result)

            cell_number += 1
        

    return list_info


list_info = ReadCell(arr_contours, gray_image)
print(list_info)



### folgende Codes stehen für Vergleich zwischen den vier Schwellenwertmethden zur Verfügung.

import cv2
import matplotlib.pyplot as plt
import os

### Quelle: https://de.wikipedia.org/wiki/Schwellenwertverfahren ###
    
### thresh1 ---> globalen Schwellenwertverfahren ###
'''
Die Hauptidee besteht darin, einen Schwellenwert festzulegen, die Pixel unterhalb des Schwellenwerts 
werden auf 0 (schwarz) und die Pixel oberhalb des Schwellenwerts auf 255 (weiß) gesetzt.
'''
### thresh2 ---> OTSU, ein verbessert globales Schwellenwertverfahren ###
'''
Eine Methode, die Schwellwerte basierend auf Bildpixeln automatisch berechnen kann.
'''
### thresh3 und thresh4 ---> lokalen Schwellenwertverfahren ###
'''
Beim lokalen Schwellenwertverfahren wird das Ausgangsbild in Regionen eingeteilt 
und der Schwellenwert für jede Region getrennt festgelegt.

Anbei gibt es zwei Methode den Schwellenwert für jede Region festzustellen:
1. Der Gaußsche gewichtete Summenalgorithmus berechnet den Abstand vom Mittelpunkt, 
   indem er die Pixel um den Mittelpunkt (x, y) der Region gemäß der Gaußschen Funktion gewichtet. ---> thresh3
2. Bei der Mittelwertmethode wird der Mittelwert des Grauwerts der Pixel im Bereich als Grauwert 
   aller Pixel im Bereich berechnet. Dies ist eigentlich ein Glättungs- oder Filtereffekt. ---> thresh4
'''


def Binar(dir):

    file_list = GetFileList(dir)
    #print(file_list)
    i = 0
    list_bina_images = []
    for image_path in file_list:
        # load the image
        img_gray = cv2.imread(dir+'\\'+image_path, 0)
        ### 1 or cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
        ### 0 or cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
        ### -1 or cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel
        
        ret,thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh3 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
        thresh4 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)


        ## hier können die Parameter 9 und 5 vielleicht durch machine learning verbessert werden

        ### folgende sind die codes dafür, alle image in einem Bild zu zeigen, um miteinander zu vergleichen.

        if __name__ == '__main__':
            titles = ['original','globalen Schwellenwertverfahren', 'OSTU', 'Gauss-Lokal', 'Mean-Lokal']
            images = [img_gray, thresh1, thresh2, thresh3, thresh4]
            zeile_of_images = len(file_list)
            plt.subplot(zeile_of_images, 5, 5*zeile_of_images)


            while i < zeile_of_images:
                for spalt in range(5):
                    plt.subplot(zeile_of_images, 5, 5*i+spalt+1), plt.imshow(images[spalt], 'gray')
                    if i == 0:
                        plt.title(titles[spalt])
                    
                    plt.xticks([]),plt.yticks([])
                break
            i+=1
        else:
            list_bina_images.append(thresh3)

    if __name__ == '__main__':
        plt.show()

    else:
        return list_bina_images  # return the threshmethode we want

    



def GetFileList(dir): # get all the filename of images unter a dir
    file_list = []

    
    file_list = os.listdir(dir) # if the input is a dir then get all the name of the files unter the dir
        # print(file_list)

    return file_list

if __name__ =='__main__':
    Binar('Development\imageTest')




import cv2
import numpy as np
import matplotlib.pyplot as plt

def Binatization(path):

    # load the image
    img_gray = cv2.imread(path, 0)

    # img = cv2.imread(path)
    # gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    cv2.imshow("img_gray", img_gray)
    cv2.waitKey()
    # print(type(img_gray))

    
    ### thresh1 und 2###
    # Die Hauptidee besteht darin, einen Schwellenwert festzulegen, die Pixel unterhalb des Schwellenwerts 
    # werden auf 0 (schwarz) und die Pixel oberhalb des Schwellenwerts auf 255 (weiß) gesetzt. Sehen wir uns 
    # den folgenden Code an.

    # Die Methode ist relativ einfach und hat eine hohe Recheneffizienz, aber es 
    # gibt ein Problem: Wenn Sie einen Stapel von Textbildern stapelweise verarbeiten möchten, 
    # einige Bilder dunklere Texte und einige Bilder hellere Texte haben, dann legen Sie einen 
    # einzelnen Schwellenwert fest und dort Das Problem ist, dass bei einem kleinen Schwellenwert 
    # der Inhalt von Text und Bildern mit heller Farbe nach der Binarisierung verloren geht.Wenn 
    # der Schwellenwert groß ist, ist der Textinhalt nach der Binarisierung des Bildes wahrscheinlich 
    # schwer zu unterscheiden aus dem Hintergrund.

    ### thresh3 ###
    # Eine Methode, die Schwellwerte basierend auf Bildpixeln automatisch berechnen kann

    # Obwohl dieser Operator die oben genannten Probleme bis zu einem gewissen Grad lösen 
    # kann, ist die Wirkung dieses Operators für eine komplexere Bildverarbeitung, insbesondere 
    # für einige Bilder mit großem Unterschied zwischen Hell und Dunkel, immer noch sehr unbefriedigend

    ### thresh4 ###
    # Um das oben erwähnte Problem zu lösen, können wir die cv2.adaptiveThreshold-Methode für die 
    # Binarisierungsverarbeitung verwenden.Die allgemeine Bedeutung der Funktion besteht darin, jedes 
    # Pixel im Bild als Zentrum zu nehmen und die Pixel im Bereich von n*n zu nehmen um ihn herum, und
    # dann wird gemäß dem Pixelwert in diesem Bereich ein Schwellenwert berechnet, um zu bestimmen, ob das 
    # aktuelle Pixel als 0 oder 255 verarbeitet wird.

    ret,thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh4 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
    #cv2.imshow('s', thresh1)
    #cv2.waitKey()
    
    
    titles = ['img', 'BINARY', 'BINARY_INV', 'BINARY_OTSU', 'BINARY_GAUSSIAN']
    images = [img_gray, thresh1, thresh2, thresh3, thresh4]
    for i in range(5):
        plt.subplot(1,5,i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()



    ### thresh3 normalerweise besser, es ist schwer eine geeignet Parameter bie thresh4 zu finden, aber wenn gefunden, ist thresh4 best.
    
    return img_gray



def GetROI(img):
    # get ROI zone

    '''
    maybe this function will be unsed for table extraction after determining the table position ###
    '''

    zone = np.ones((200, 100, 1)) # define a 200*100 matrix, 1 mains 1 channel

    zone = img[200:400, 200:300] # write grayscale values into matrix, long from 200 to 400, hight from 200 to 300

    cv2.imshow("Zone", zone)
    cv2.waitKey(0)

    # fusion
    img[0:200, 0:100] = zone




Binatization('Development\imageTest\image1.png')
Binatization('Development\imageTest\image2.png')

 



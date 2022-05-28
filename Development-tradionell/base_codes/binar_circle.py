import cv2
import matplotlib.pyplot as plt
import os


'''
Binarization --> Gauss_Filter --> BInarization
'''

def Circle1(path, circle_time):   # Linienbreite vergrößert sich
    gray_image = cv2.imread(path, 0)
    i = 1
    while i <= circle_time:
        bina_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5) # thresh3
        filter_image = cv2.GaussianBlur(bina_image, (3,3), 0) # gaussblur
        bina_image = cv2.adaptiveThreshold(filter_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
        gray_image = bina_image # um circle zu machen
        i += 1
    
    
    cv2.imshow('',bina_image)
    cv2.waitKey()
    return bina_image

# Circle1('Development\imageTest\image1.png', 2)

# Jeder Zyklus macht es schlimmer bei Circle1

def Circle2(path, circle_time): # Rauschunterdrückung
    gray_image = cv2.imread(path, 0)
    i = 1
    while i <= circle_time:
        bina_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5) # thresh3
        filter_image = cv2.GaussianBlur(bina_image, (3,3), 0) # gaussblur
        ret, bina_image = cv2.threshold(filter_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # OSTU
        gray_image = bina_image # um circle zu machen
        i += 1
    
    
    cv2.imshow('',bina_image)
    cv2.waitKey()
    return bina_image


if __name__ == '__main__':
    # Circle1('Development\imageTest\image1.png', 2)
    Circle2(r'Development\imageTest\rotate_table.png', 5) # erhöhen das Parameter wird die Leistung verbessert.

# Jede Schleife erhöht die Anzahl der weißen Pixel und konvergiert schließlich zu einem Wert.

# der Grund für die Unterschiede zwischen Circle1 und 2 lieht vielleicht in ***Schwelle***



        


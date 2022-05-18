'''
- ROI of Zelle in table
- erode then dilate ---> reduce the noise
   
    ----------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    bina_image = ~cv2.erode(~bina_image, kernel,iterations = 1)   # erode to noise reduction
    # erode should after ROI before Tesseract
    ----------------
    
- Get Words in Zelle
    - Methode 1. Textfokus: https://wenku.baidu.com/view/8bdffcf175a20029bd64783e0912a21614797fd3.html
    - Methode 2. Tesseract
- write Words in JISON
- JISON in Excal


'''

import pytesseract
import cv2
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


gray_image = cv2.imread(r'Development\imageTest\einfach_table.jpg', 0)
bina_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)

cv2.imshow('', bina_image)
cv2.waitKey()

result = pytesseract.image_to_string(bina_image)
print(type(result))
result.replace(' ', '\n')
re = result.split(' ')
print(re)

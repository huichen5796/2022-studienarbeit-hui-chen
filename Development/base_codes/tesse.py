### how to install: https://www.jianshu.com/p/93ab58dea50f ###
### https://github.com/tesseract-ocr/tessdoc/blob/main/Downloads.md 
import pytesseract
from pro_processing import Circle2
import cv2


# bina_image = Circle2('Development\\imageTest\\test.png', 5)

bina_image = cv2.imread('Development\\imageTest\\test.png')

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
result = pytesseract.image_to_string(bina_image)
print(result)

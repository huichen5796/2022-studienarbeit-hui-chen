import pytesseract
import easyocr
import cv2
import matplotlib.pyplot as plt

def Extrakt_Tesseract(image_cell):

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    result = pytesseract.image_to_string(image_cell)
    # print(result)

    if '\n' in result:
        result = result.replace('\n', '')
    return result


def Extrakt_Esayocr(image_cell): # for this function can image_cell auch path is

    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_cell)[0][1]
    
    return result

if __name__ == '__main__':
    
    reader = easyocr.Reader(['en'])

    img = cv2.imread('Development\imageTest\einfach_table.jpg')
    result = reader.readtext(img)
    color = (0,0,255)
    thick = 3
    for res in result:
        print(res)
        pos = res[0]
        text = res[1]
        for p in [(0,1), (1,2), (2,3), (3,0)]:
            cv2.line(img, pos[p[0]], pos[p[1]], color, thick)
    plt.imshow(img, cmap = 'gray')
    plt.xticks([]),plt.yticks([])
    plt.show()
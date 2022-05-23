import pytesseract
import easyocr

def Extrakt_Tesseract(image_cell):

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    result = pytesseract.image_to_string(image_cell)
    # print(result)

    if '\n' in result:
        result = result.replace('\n', '')
    return result


def Extrakt_Esayocr(image_cell):

    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_cell)

    return result
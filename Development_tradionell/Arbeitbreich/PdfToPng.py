

### hier kann alle Images unter a dir ausgenommen werden ----- durch ImageReformat() --- in 'image_list'
### dabei wird auch alle PDFs in 'Development\PDF' umgezogen, 
### Jede Seite der PDF-Datei wird in ein PNG-Bild konvertiert und hier gespeichert ----- 'Development\imageTest'

import fitz
import os
import shutil


def PdfToPng(pdf_path, save_path):
    #### change all PDFs unter dir to PNG
    
    doc = fitz.open(pdf_path)
    for pg in range(doc.pageCount):
        page = doc[pg]
        rotate = int(0)
        zoom_x = 2.0
        zoom_y = 2.0
        trans = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
        pm = page.get_pixmap(matrix= trans, alpha = False)
        pm.save(save_path +'\\'+ os.path.basename(pdf_path)[:-4] + '_%s.png' % pg)
        # save_path + os.path.basename(pdf_path)[:-4]
    
    print('%s has %d pages' %(os.path.basename(pdf_path), pg+1))
        


def PDFRemover(dir_name):   
    # remove the PDFs unter dir_name to 'Development\PDF'
    n = 0
    file_list = os.listdir(dir_name)
    for i in range(len(file_list)):
        file1 = file_list[i]
        if os.path.splitext(file1)[1] in ['.pdf', '.PDF']:
            shutil.copy(dir_name +'\\'+ file1, 'Development_tradionell\PDF')
            os.remove(dir_name +'\\'+ file1) 
            n += 1
        else:
            continue
    print("darunter gibt es %s PDFs, entfernt zu 'Development_tradionell\PDF'." %n)


def GetImageList(dir_name): 
    ## https://blog.csdn.net/weixin_39633252/article/details/110518896
 
    # get all the filename of images unter a dir
    
    file_list = os.listdir(dir_name) # if the input is a dir then get all the name of the files unter the dir
    
    return file_list
    

def ImageReformat(dir):
    print('---------------------------')
    PDFRemover(dir)
    pdf_list = GetImageList('Development_tradionell\PDF')
    for pdf in pdf_list:
        pdf_path = 'Development_tradionell\PDF' + '\\' + pdf
        save_path = 'Development_tradionell\imageTest'
        PdfToPng(pdf_path, save_path)

    image_list = GetImageList(dir)
    print('There are %s images in total, including the images from the pdfs.'%len(image_list))
    print('---------------------------')
    # print(image_list)
    return image_list

if __name__ == '__main__':
    ImageReformat('Development_tradionell\imageTest')
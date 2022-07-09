
import os
import fitz
import shutil
from functions import Main, Search
from elasticsearch import Elasticsearch
es = Elasticsearch()


def GetImageList(dir_name):
    '''
    get all the filename of images unter a dir

    - input: path of dir

    - output: all the files unter the dir, in form list 

    '''

    try:
        file_list = os.listdir(dir_name)
        # if the input is a path of dir then get all the name of the files unter the dir

        return file_list

    except Exception as e:
        print('ERROR BY GetImageList: ' + str(e))


def PDFRemover(dir_name):
    '''
    remove the PDFs unter dir_name to 'Development\\PDF'

    - input: the path of directory

    - output: None

    '''
    try:
        n = 0
        file_list = GetImageList(dir_name)
        for file in file_list:

            if os.path.splitext(file)[1] in ['.pdf', '.PDF']:
                shutil.copy(dir_name + '\\' + file, 'Development\\PDF')
                os.remove(dir_name + '\\' + file)
                n += 1

            else:
                continue
        print("there are %s PDFs, removed to 'Development\\PDF'." % n)

    except Exception as e:
        print('ERROR BY PDFRemover: ' + str(e))


def PdfToPng(pdf_path, save_path):
    '''
    change all PDFs unter dir to PNG

    - input 1: the path of pdf
    - input 2: the path to save the pdf

    - output: None

    '''
    try:
        doc = fitz.open(pdf_path)
        for pg in range(doc.pageCount):  # pg ist die Seitenummer
            page = doc[pg]
            rotate = int(0)
            zoom_x = 2.0
            zoom_y = 2.0
            trans = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
            pm = page.get_pixmap(matrix=trans, alpha=False)
            pm.save(save_path + '\\' +
                    os.path.splitext(os.path.basename(pdf_path))[0] + '_%s.png' % pg)

        print('%s has %d pages' % (os.path.basename(pdf_path), pg+1))
    except:
        print('ERROR BY PdfToPng OF %s' % (pdf_path))


def ImageReformat(dir):
    '''
    - input: path of dir

    - output: image_list

    hier kann alle Images unter a dir vorverarbeitet werden ----- durch ImageReformat() --- in 'image_list'
    dabei wird alle PDFs in 'Development\PDF' umgezogen, 
    Jede Seite der PDF-Datei wird in ein PNG-Bild konvertiert und hier gespeichert ----- 'Development\imageTest'
    '''
    try:
        PDFRemover(dir)
        pdf_list = GetImageList('Development\\PDF')
        for pdf in pdf_list:
            pdf_path = 'Development\\PDF' + '\\' + pdf
            save_path = 'Development\\imageTest'
            PdfToPng(pdf_path, save_path)

        image_list = GetImageList(dir)
        print('There are %s images in total, including the images from the pdfs.' % len(
            image_list))
        print('---------------------------')
        # print(image_list)
        return image_list
    except:
        print('ERROR BY ImageReformat')


def StapelVerbreitung(dir):

    image_list = GetImageList(dir)
    print('---------------------------')
    print('There are %s images in total. -- include unprocessed Pdfs.' %
          len(image_list))
    image_list = ImageReformat(dir)

    # print(image_list)
    path_images = [os.path.normpath(os.path.join(dir, fn))
                   for fn in image_list]
    for image in path_images:
        Main(image)
    print('done')


if __name__ == '__main__':
    es.indices.delete(index='table', ignore=[400, 404])  # deletes whole index
    StapelVerbreitung('Development\\imageTest')

    result = Search('table', 'all')
    print(result)

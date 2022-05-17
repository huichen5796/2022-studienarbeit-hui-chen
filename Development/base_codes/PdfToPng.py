import fitz
import os




def PdfToPng(path):
    #### change all PDFs unter dir to PNG

    doc = fitz.open(path)
    for pg in range(doc.pageCount):
        page = doc[pg]
        rotate = int(0)
        zoom_x = 2.0
        zoom_y = 2.0
        trans = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
        pm = page.get_pixmap(matrix= trans, alpha = False)
        pm.save(os.path.splitext(path)[0] + '_%s.png' % pg)


def GetFileList(dir_name): 
    file_list = []
    png_list = []
    pdf_list = []
    #print(pdf_list)
    # get all the filename of images unter a dir
    
    file_list = os.listdir(dir_name) # if the input is a dir then get all the name of the files unter the dir
    
    # print(file_list)
    #print(png_list)
    #print(pdf_list)

    for file in file_list:
        if os.path.splitext(file)[1] in ['.png', '.PNG']:
            png_list.append(file)
            # print(os.path.splitext(file)[1])
        elif os.path.splitext(file)[1] in ['.pdf', '.PDF']:
            pdf_list.append(file)

            path1 = os.path.abspath(file)
            print(path1)
            PdfToPng(path1)

    #print(png_list)
    #print(pdf_list)
    print(file_list)
    return file_list

GetFileList('Development\imageTest')
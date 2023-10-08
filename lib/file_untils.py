import os
import fitz
import time
import shutil
import datetime

def get_image_list(dir_name):
    file_list = os.listdir(dir_name)
    return file_list

def open_all(dir_name):
    file_list = get_image_list(dir_name)
    for file in file_list:
        if os.path.splitext(file)[1] in ['.pdf', '.PDF']:
            pdf_path = dir_name + '/' + file
            save_path = dir_name
            pdf_to_png(pdf_path, save_path)
            if not os.path.exists('PDFs'):
                os.makedirs('PDFs')
            shutil.copy(dir_name + '/' + file, 'PDFs')
            os.remove(dir_name + '/' + file)
    return get_image_list(dir_name)

def pdf_to_png(pdf_path, save_path):
    doc = fitz.open(pdf_path)
    print('%s has %d pages' % (os.path.basename(pdf_path), doc.page_count))
    start = time.perf_counter()
    for pg in range(doc.page_count):  # pg ist die Seitenummer
        page = doc[pg]
        rotate = int(0)
        zoom_x = 2.0
        zoom_y = 2.0
        trans = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
        pm = page.get_pixmap(matrix=trans, alpha=False)
        pm.save(save_path + '/' +
                os.path.splitext(os.path.basename(pdf_path))[0] + '_%s.png' % pg)

        finish = '▓' * (pg+1)
        need_do = '-' * (doc.page_count-pg-1)
        dur = time.perf_counter() - start
        if pg == doc.page_count-1:
            print("\r{}/{}|{}{}|{:.2f}s".format((pg+1),
                    doc.page_count, finish, need_do, dur))
        else:
            print("\r{}/{}|{}{}|{:.2f}s".format((pg+1),
                    doc.page_count, finish, need_do, dur), end='')

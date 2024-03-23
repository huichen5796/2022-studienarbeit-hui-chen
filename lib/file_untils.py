import os
import fitz
import time
import shutil
import cv2
import datetime

FROM = 'store_ori_file'
DOING = 'store_image_finder'

def get_file_list(dir_name):
    file_list = os.listdir(dir_name)
    return file_list

def get_all_images_in_folder(folder_path):
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                file_path = os.path.join(root, file)
                image_paths.append(file_path)
    return image_paths

def open_all(dir_name = FROM, result_dir = DOING):
    file_list = get_file_list(dir_name)
    for file in file_list:
        if os.path.splitext(file)[1] in ['.pdf', '.PDF']:
            pdf_path = dir_name + '/' + file
            save_path = result_dir + '/' + file[0:-4]
            os.makedirs(save_path)
            pdf_to_png(pdf_path, save_path)
        else:
            shutil.copy(dir_name + '/' + file, result_dir)
    return get_all_images_in_folder(result_dir)

def pdf_to_png(pdf_path, save_path):
    doc = fitz.open(pdf_path)
    # print('%s has %d pages' % (os.path.basename(pdf_path), doc.page_count))
    # start = time.perf_counter()
    for pg in range(doc.page_count):  # pg ist die Seitenummer
        page = doc[pg]
        rotate = int(0)
        zoom_x = 2.0
        zoom_y = 2.0
        trans = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
        pm = page.get_pixmap(matrix=trans, alpha=False)
        pm.save(save_path + '/' +
                os.path.splitext(os.path.basename(pdf_path))[0] + '_%s.png' % pg)

        # finish = '▓' * (pg+1)
        # need_do = '-' * (doc.page_count-pg-1)
        # dur = time.perf_counter() - start
        # if pg == doc.page_count-1:
        #     print("\r{}/{}|{}{}|{:.2f}s".format((pg+1),
        #             doc.page_count, finish, need_do, dur))
        # else:
        #     print("\r{}/{}|{}{}|{:.2f}s".format((pg+1),
        #             doc.page_count, finish, need_do, dur), end='')
            

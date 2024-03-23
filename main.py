from pipeline_3 import pipeline_3

from lib import *
from conf import *

import os
import time
import shutil

def main(path = 'store_image_finder'):
    if os.path.isdir(path):
        images = open_all(path)
        path_images = [os.path.normpath(os.path.join(path, fn)) for fn in images]
        start = time.perf_counter()
        n = 0
        for i, image_path in enumerate(path_images):
            try:
                pipeline_3(image_path)
            except:
                n += 1

            finish = '▓' * int((i+1)*(50/len(path_images)))
            need_do = '-' * (50-int((i+1)*(50/len(path_images))))
            dur = time.perf_counter() - start
            if i == len(path_images)-1:
                print("\r{}/{}|{}{}|{:.2f}s".format((i+1), len(path_images), finish, need_do, dur) +
                    ' done: ' + os.path.basename(image_path)+' error: %s, finish' % n, flush=True)
            else:
                print("\r{}/{}|{}{}|{:.2f}s".format((i+1), len(path_images), finish, need_do, dur) +
                    ' done: ' + os.path.basename(image_path)+' error: %s' % n, end='', flush=True)
                
    else:
        n = 0
        start = time.perf_counter()
        try:
            pipeline_3(path)
        except:
            n = 1
        finish = '▓' * 50
        need_do = '-' * 0
        dur = time.perf_counter() - start
        print("\r{}/{}|{}{}|{:.2f}s".format((1), 1, finish, need_do, dur) +
                    ' done: ' + os.path.basename(path)+' error: %s' % n, end='', flush=True)


if __name__ == '__main__':
### for test:
    # print(ElasticUntils('table').save_excel(saveRoot='excels', tableId='all'))
    # print(ElasticUntils('table').save_excel(saveRoot='excels', imageId="test2.PNG"))
    # print(ElasticUntils('table', os.path.basename(FILE_PATH), 0).search())
    # print(ElasticUntils('table').search(search_all=True))
    # ElasticUntils('table', os.path.basename(FILE_PATH), 0).detele()
    # ElasticUntils('table').detele(delete_all=True)
    
    # main('Abbildungen/test2.PNG')
    main('successControl')
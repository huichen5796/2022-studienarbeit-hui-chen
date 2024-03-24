from pipeline_3 import pipeline_3

from lib import *
from conf import *

import os
import time
import shutil

log_writer = LogWriter()

def main(path = 'store_image_finder'):
    path_images = open_all(path)
    start = time.perf_counter()
    for i, image_path in enumerate(path_images):
        error = ''
        try:
            pipeline_3(image_path)
            log_writer.writeSuccess(f'DONE: {image_path}')
        except Exception as e:
            log_writer.writeError(str(e))
            error = str(e)

        finish = '▓' * int((i+1)*(50/len(path_images)))
        need_do = '-' * (50-int((i+1)*(50/len(path_images))))
        dur = time.perf_counter() - start
        if i == len(path_images)-1:
            print("\r{}/{}|{}{}|{:.2f}s".format((i+1), len(path_images), finish, need_do, dur) +
                ' done: ' + os.path.basename(image_path)+' error: %s, finish' % error, flush=True)
        else:
            print("\r{}/{}|{}{}|{:.2f}s".format((i+1), len(path_images), finish, need_do, dur) +
                ' done: ' + os.path.basename(image_path)+' error: %s' % error, end='', flush=True)


if __name__ == '__main__':
### for test:
    # print(ElasticUntils('table').save_excel(saveRoot='excels', tableId='all'))
    # print(ElasticUntils('table').save_excel(saveRoot='excels', imageId="test2.PNG"))
    # print(ElasticUntils('table', os.path.basename(FILE_PATH), 0).search())
    # print(ElasticUntils('table').search(search_all=True))
    # ElasticUntils('table', os.path.basename(FILE_PATH), 0).detele()
    # ElasticUntils('table').detele(delete_all=True)
    
    main('Abbildungen/test2.PNG')
    # main('successControl')
import os
import cv2
import shutil


root_image = "Grundlagen_Erkenntinesse\\ML\\Modelle\\marmot\\image_v2\\"
root_mask = "Grundlagen_Erkenntinesse\\ML\\Modelle\\marmot\\table_mask\\"

#root_image_val = "Development_DL\\Arbeitbreich_DL\\marmot_new\\image_val\\"
#root_mask_val = "Development_DL\\Arbeitbreich_DL\\marmot_new\\table_mask_val\\"

imgs_list = [pp for pp in os.listdir(root_image)]
masks_list = [pp for pp in os.listdir(root_mask)]

# check
m = 0
'''
for i in range(0, len(masks_list)):
    if os.path.splitext(masks_list[i])[1] == '.png':
        os.remove(root_mask + masks_list[i])

'''

for i in range(0, len(imgs_list)):

    if imgs_list[i] == masks_list[i].replace('_table_mask.jpg','.jpg'):
        continue
    else:
        print((imgs_list[i], masks_list[i]))
if m == 0:
    print('same')


'''
for fn in masks_list:
        
    img_path = os.path.join(root_image_val, fn.replace('_table_mask.jpg','.jpg'))
    mask_path = os.path.join(root_mask_val, fn)
        
    mask = cv2.imread(mask_path, 0)
    c, h = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if len(c) == 0:
        continue
    else:
        shutil.copy(img_path, 'Development_DL\Arbeitbreich_DL\marmot_new\\image_val')
        shutil.copy(mask_path, 'Development_DL\Arbeitbreich_DL\marmot_new\\table_mask_val')

'''
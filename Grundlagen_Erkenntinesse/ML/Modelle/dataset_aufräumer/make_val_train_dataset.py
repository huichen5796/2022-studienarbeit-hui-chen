import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms.functional import to_tensor
from torchvision.io import read_image
import numpy as np
import shutil
from PIL import Image
import cv2



def GetVal():    # Do not run
    # the shape of image is (1024, 1024)
    # class the dataset to training set and validation set

    root_image = "Development_DL\\Arbeitbreich_DL\\marmot\\image\\"
    root_mask = "Development_DL\\Arbeitbreich_DL\\marmot\\table_mask\\"

    root_image_val = "Development_DL\\Arbeitbreich_DL\\marmot\\image_val\\"
    root_mask_val = "Development_DL\\Arbeitbreich_DL\\marmot\\table_mask_val\\"

    # get test set

    images_list = [pp for pp in os.listdir(root_image)]
    masks_list = [pp for pp in os.listdir(root_mask)]

    for m in masks_list:
        if not m.endswith('.jpg'):
            os.remove(root_mask+m)

    path_images = [os.path.join(root_image, fn) for fn in images_list]
    path_masks = [os.path.join(root_mask, fm) for fm in masks_list]

    np.random.seed(2)
    val_rnd_imgs = np.random.choice(images_list, 200, replace = False)
    np.random.seed(2)
    val_rnd_masks = np.random.choice(masks_list, 200, replace = False)

    #print((val_rnd_imgs))
    #print(val_rnd_masks)
    
    for image in val_rnd_imgs:
        shutil.copy(root_image+image, root_image_val+image)
        os.remove(root_image+image)

    for mask in val_rnd_masks:
        shutil.copy(root_mask+mask, root_mask_val+mask)
        os.remove(root_mask+mask)





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

    root_image = "mardata\\image\\"
    root_mask = "mardata\\table_mask\\"

    root_image_val = "mardata\\image_val\\"
    root_mask_val = "mardata\\table_mask_val\\"

    # get test set

    images_list = [pp for pp in os.listdir(root_image)]
    masks_list = [pp.replace('.jpg', '_table_mask.jpg') for pp in images_list]


    np.random.seed(2)
    val_rnd_imgs = np.random.choice(images_list, 200, replace = False)
    val_rnd_masks = [pp.replace('.jpg', '_table_mask.jpg') for pp in val_rnd_imgs]
    
    for image in val_rnd_imgs:
        shutil.copy(root_image+image, root_image_val+image)
        os.remove(root_image+image)

    for mask in val_rnd_masks:
        shutil.copy(root_mask+mask, root_mask_val+mask)
        os.remove(root_mask+mask)


# GetVal()


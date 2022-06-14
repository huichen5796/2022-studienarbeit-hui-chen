import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
import os
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import cv2


def ShowData():
    # data path
    path_image = "Development_DL\\Arbeitbreich_DL\\marmot\\image\\"
    path_mask = "Development_DL\\Arbeitbreich_DL\\marmot\\table_mask\\"

    imgs_list = [pp for pp in os.listdir(path_image)]
    masks_list = [pp for pp in os.listdir(path_mask)]
    print("number of images:", len(imgs_list))
    print("number of masks:", len(masks_list))
    """
    number of images: 993
    number of masks: 993
    """

    # show some image
    np.random.seed(1)
    rnd_imgs = np.random.choice(imgs_list, 4)
    print('The random images are: ', rnd_imgs)
    # The random images are:  ['10.1.1.1.2028_8.jpg', '10.1.1.13.2927_6.jpg', '10.1.1.1.2064_5.jpg', '10.1.1.8.2176_133.jpg']

    def show_img_mask(img, mask):
        if torch.is_tensor(img):
            img = to_pil_image(img)
            mask = to_pil_image(mask)
            
        img_mask = mark_boundaries(
                    np.array(img), 
                    np.array(mask),

                )
        plt.imshow(img_mask)

    i = 0
    plt.figure(figsize=(10, 10))
    for fn in rnd_imgs:
        
        img_path = os.path.join(path_image, fn)
        mask_path = os.path.join(path_mask, fn.replace('.jpg', '_table_mask.jpg'))
        
        img = cv2.imread(img_path, 0)
        mask = cv2.imread(mask_path, 0)

        print('shape of image is: ',img.shape)
        print('shape of mask is: ',mask.shape)   


        plt.subplot(4, 3, 3*i+1) 
        plt.imshow(img, cmap="gray")

        plt.subplot(4, 3, 3*i+2) 
        plt.imshow(mask, cmap="gray")

        plt.subplot(4, 3, 3*i+3) 
        show_img_mask(img, mask)
        i+=1

    plt.show()

ShowData()



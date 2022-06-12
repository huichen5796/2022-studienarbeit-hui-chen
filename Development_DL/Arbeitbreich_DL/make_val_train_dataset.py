import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms.functional import to_tensor
from torchvision.io import read_image
import numpy as np
import shutil
from PIL import Image
import cv2



def Aufraeumen():    # Do not run
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



# make Dataset class
class MakeDataset(Dataset):
    def __init__(self, root_image1, root_mask1):
        images_list = [pp for pp in os.listdir(root_image1)]
        masks_list = [pp for pp in os.listdir(root_mask1)]
        self.path_images = [os.path.join(root_image1, fn) for fn in images_list]
        self.path_masks = [os.path.join(root_mask1, fn) for fn in masks_list]

        
    def __len__(self):
        return len(self.path_images)
    
    def __getitem__(self, idx):
        path_image = self.path_images[idx]
        image = cv2.imread(path_image,0)
        image = cv2.resize(image, (512, 512))
        
        path_mask = self.path_masks[idx]
        mask = cv2.imread(path_mask,0)
        mask = cv2.resize(mask, (512, 512))


        
        image = to_tensor(image)
        mask = to_tensor(mask)

        return image, mask


def GetTrainVal():
    root_image = "Development_DL\\Arbeitbreich_DL\\marmot\\image\\"
    root_mask = "Development_DL\\Arbeitbreich_DL\\marmot\\table_mask\\"

    root_image_val = "Development_DL\\Arbeitbreich_DL\\marmot\\image_val\\"
    root_mask_val = "Development_DL\\Arbeitbreich_DL\\marmot\\table_mask_val\\"

    # instantiate dataset
    train_ds = MakeDataset(root_image, root_mask)
    val_ds = MakeDataset(root_image_val, root_mask_val)


    print('The total number of images in the train dataset: ',len(train_ds))
    print('The total number of images in the validation dataset: ',len(val_ds))





    # make dataloader
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=True)

    # see the infos of datas
    for img, mask in train_dl:
        print('every batch of train image: ',img.shape, img.dtype)
        # torch.Size([8, 3, 1024, 1024]) torch.float32
        print('every batch of train mask: ',mask.shape, mask.dtype)
        # torch.Size([8, 3, 1024, 1024]) torch.float32
        break

    for img, mask in val_dl:
        print('every batch of val image: ',img.shape, img.dtype)
        # torch.Size(16, 3, 1024, 1024]) torch.float32
        print('every batch of val mask: ',mask.shape, mask.dtype)
        # torch.Size([16, 3, 1024, 1024]) torch.float32
        break

    return train_dl, val_dl

# GetTrainVal()
'''
The total number of images in the train dataset:  793
The total number of images in the validation dataset:  200
every batch of train image:  torch.Size([8, 1, 512, 512]) torch.float32
every batch of train mask:  torch.Size([8, 1, 512, 512]) torch.float32
every batch of val image:  torch.Size([16, 1, 512, 512]) torch.float32
every batch of val mask:  torch.Size([16, 1, 512, 512]) torch.float32

'''

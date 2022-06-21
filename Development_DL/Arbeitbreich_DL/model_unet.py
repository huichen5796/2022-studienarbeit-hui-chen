#https://github.com/pranjalrai-iitd/Fetal-head-segmentation-and-circumference-measurement-from-ultrasound-images/blob/master/Unet.py
#https://www.jianshu.com/p/7086ded792b2
#https://github.com/codecat0/CV/blob/main/Semantic_Segmentation/SegNet/nets/segent.py
#https://medium.com/analytics-vidhya/table-extraction-using-deep-learning-3c91790aa200

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
import cv2
import os 
import copy
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np



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
        image = cv2.resize(image, (1024, 1024))
        ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #image /= 255.0 # Normalisierung

        path_mask = self.path_masks[idx]
        mask = cv2.imread(path_mask,0)
        mask = cv2.resize(mask, (1024, 1024))
        ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #mask /= 255.0

        image = image.astype('uint8')
        mask = mask.astype('uint8') 
        
        image = to_tensor(image)
        mask = to_tensor(mask)

        return image, mask


def GetTrainVal(batch_size):
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
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=True)

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

    return train_dl, val_dl, len(train_ds),len(val_ds)

#GetTrainVal()
'''
The total number of images in the train dataset:  793
The total number of images in the validation dataset:  200
every batch of train image:  torch.Size([8, 1, 1024, 1024]) torch.float32
every batch of train mask:  torch.Size([8, 1, 1024, 1024]) torch.float32
every batch of val image:  torch.Size([16, 1, 1024, 1024]) torch.float32
every batch of val mask:  torch.Size([16, 1, 1024, 1024]) torch.float32

'''

# encoder-decoder model  U-Net

class conv_block(nn.Module):

    def __init__(self, input_channels, output_channels, down=True):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride = 1, padding=1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace = True),
                                   nn.Dropout(0.5),

                                   nn.Conv2d(output_channels, output_channels, kernel_size=3, stride = 1, padding=1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace = True),
                                   nn.Dropout(0.5),
                                  )

    def forward(self, x):
        x = self.conv(x)

        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self,x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(img_ch,64)
        self.Conv2 = conv_block(64,128)
        self.Conv3 = conv_block(128,256)
        self.Conv4 = conv_block(256,512)
        self.Conv5 = conv_block(512,1024)

        self.Up5 = up_conv(1024,512)
        self.Up_conv5 = conv_block(1024, 512)

        self.Up4 = up_conv(512,256)
        self.Up_conv4 = conv_block(512,256)
        
        self.Up3 = up_conv(256,128)
        self.Up_conv3 = conv_block(256,128)
        
        self.Up2 = up_conv(128,64)
        self.Up_conv2 = conv_block(128,64)

        self.Conv_1x1 = nn.Sequential(nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0),
                                      nn.Sigmoid()                      
                                      )



    def forward(self,x):

        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


# GPU
device = 'cuda' 


def dice_sim(pred, truth): # to judge Änlichkeit zwischen mask und pred
    epsilon = 1e-8
    num_batches = pred.size(0)
    m1 = pred.view(num_batches, -1).bool()
    m2 = truth.view(num_batches, -1).bool()

    intersection = torch.logical_and(m1, m2).sum(dim=1)
    return (((2. * intersection + epsilon) / (m1.sum(dim=1) + m2.sum(dim=1) + epsilon)).sum(dim=0))/2

num_epochs = 50
batch_size = 8

train_dl, val_dl, train_size, val_size = GetTrainVal(batch_size)

model = U_Net().to(device)

# helper functions
loss_fn = nn.BCELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, min_lr=1e-9)
train_size = train_size
val_size = val_size
print('--------------------------------start--------------------------------------')
for epoch in range(1,num_epochs+1):
    print('---------------------------------------------------------------------------')
    print('epoch %s' % epoch)

    running_loss = []
    val_loss = []
    val_acc = []

    epoch_train_loss_list = []
    epoch_val_loss_list = []
    epoch_val_acc_list = []


    # Training
    for image, truth in train_dl:
        predictions = model(image.cuda())
        loss = loss_fn(predictions, truth.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
    # Validation
    with torch.no_grad():
        for image1, truth1 in val_dl:
            predictions = model(image.cuda())
            loss = loss_fn(predictions, truth.cuda())
            val_loss.append(loss.item())
            val_acc.append(dice_sim(predictions, truth.cuda())*100)

    epoch_train_loss = sum(running_loss) / len(running_loss)
    epoch_val_loss = sum(val_loss) / len(val_loss)
    epoch_val_acc = sum(val_acc) / len(val_acc)
    scheduler.step(epoch_val_loss) # LR Scheduler

    epoch_train_loss_list.append(epoch_train_loss)
    epoch_val_loss_list.append(epoch_val_loss)
    epoch_val_acc_list.append(epoch_val_acc)


    print(f"==>train_loss: {epoch_train_loss} ==>val_loss: {epoch_val_loss} ==>val_accuracy: {epoch_val_acc}")
    epoch += 1

#Visualize the results

plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),epoch_train_loss_list,label="train")
plt.plot(range(1,num_epochs+1),epoch_val_loss_list,label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

# plot accuracy progress
plt.title("Val Accuracy")
plt.plot(range(1,num_epochs+1),epoch_val_acc_list)
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

a = open('Development_DL\Arbeitbreich_DL\model.pkl')
a.close()
torch.save(model, 'Development_DL\Arbeitbreich_DL\model.pkl')

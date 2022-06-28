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
from albumentations import (HorizontalFlip, VerticalFlip, Compose, Resize,)
import torchvision
device = 'cuda'

# transforms
h, w = 1024,1024
transform_train = Compose([ Resize(h, w), 
                HorizontalFlip(p=0.5), 
                VerticalFlip(p=0.5), 
              ])

transform_val = Resize(h, w)

# make Dataset class
class MakeDataset(Dataset):
    def __init__(self, root_image1, root_mask1, transform = None):
        images_list = [pp for pp in os.listdir(root_image1)]
        masks_list = [pp for pp in os.listdir(root_mask1)]
        self.path_images = [os.path.join(root_image1, fn) for fn in images_list]
        self.path_masks = [os.path.join(root_mask1, fn) for fn in masks_list]
        self.transform = transform

    def __len__(self):
        return len(self.path_images)
    
    def __getitem__(self, idx):
        path_image = self.path_images[idx]
        image = cv2.imread(path_image,0)
        image = cv2.resize(image, (1024, 1024))
        ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #image = image.astype(np.float32)
        #image /= 255.0 # Normalisierung

        path_mask = self.path_masks[idx]
        mask = cv2.imread(path_mask,0)
        mask = cv2.resize(mask, (1024, 1024))
        ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #mask = mask.astype(np.float32)
        #mask /= 255.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = image.astype('uint8')
        mask = mask.astype('uint8') 
        
        image = to_tensor(image)
        mask = to_tensor(mask)

        return image, mask


def GetTrainVal(batch_size):
    root_image = "/content/drive/MyDrive/githui/marmot/image"
    root_mask = "/content/drive/MyDrive/githui/marmot/table_mask"

    root_image_val = "/content/drive/MyDrive/githui/marmot/image_val"
    root_mask_val = "/content/drive/MyDrive/githui/marmot/table_mask_val"
    

    # instantiate dataset
    train_ds = MakeDataset(root_image, root_mask, transform = transform_train)
    val_ds = MakeDataset(root_image_val, root_mask_val, transform = transform_val)

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

    return train_dl, val_dl, len(train_ds), len(val_ds)

#GetTrainVal()
'''
The total number of images in the train dataset:  793
The total number of images in the validation dataset:  200
every batch of train image:  torch.Size([8, 1, 1024, 1024]) torch.float32
every batch of train mask:  torch.Size([8, 1, 1024, 1024]) torch.float32
every batch of val image:  torch.Size([16, 1, 1024, 1024]) torch.float32
every batch of val mask:  torch.Size([16, 1, 1024, 1024]) torch.float32

'''
#---------------------------------------------------------------------------------------------------------------#
# encoder-decoder model U-Net

class conv_block(nn.Module):

    def __init__(self, input_channels, output_channels, down=True):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride = 1, padding=1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace = True),

                                   nn.Conv2d(output_channels, output_channels, kernel_size=3, stride = 1, padding=1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace = True)
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

        self.Conv1 = conv_block(img_ch,32)
        self.Conv2 = conv_block(32,64)
        self.Conv3 = conv_block(64,128)
        self.Conv4 = conv_block(128,256)
        self.Conv5 = conv_block(256,512)

        self.Up5 = up_conv(512,256)
        self.Up_conv5 = conv_block(512, 256)

        self.Up4 = up_conv(256,128)
        self.Up_conv4 = conv_block(256, 128)
        
        self.Up3 = up_conv(128,64)
        self.Up_conv3 = conv_block(128, 64)
        
        self.Up2 = up_conv(64,32)
        self.Up_conv2 = conv_block(64,32)

        self.Conv_1x1 = nn.Sequential(nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0),
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
#---------------------------------------------------------------------------------------------------------------#
# model R2U_Net
class Recurrent_block(nn.Module):

    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))


    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

class R2U_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=32,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=32,ch_out=64,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=64,ch_out=128,t=t)
               
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=128, ch_out=64,t=t)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=64, ch_out=32,t=t)

        self.Conv_1x1 = nn.Sequential(nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0),
                                      nn.Sigmoid()
                                      )


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        d3 = self.Up3(x3)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
#---------------------------------------------------------------------------------------------------------------#
#model TableNet
class VGG19(nn.Module):
    def __init__(self, pretrained = True, requires_grad = True):
        super(VGG19, self).__init__()
        _vgg = torchvision.models.vgg19(pretrained=pretrained).features
        self.vgg_pool3 = torch.nn.Sequential()
        self.vgg_pool4 = torch.nn.Sequential()
        self.vgg_pool5 = torch.nn.Sequential()

        for x in range(19):
            self.vgg_pool3.add_module(str(x), _vgg[x])
        for x in range(19, 28):
            self.vgg_pool4.add_module(str(x), _vgg[x])
        for x in range(28, 37):
            self.vgg_pool5.add_module(str(x), _vgg[x])

        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x): 
        pool_3_out = self.vgg_pool3(x) #torch.Size([1, 256, 128, 128])
        pool_4_out = self.vgg_pool4(pool_3_out) #torch.Size([1, 512, 64, 64])
        pool_5_out = self.vgg_pool5(pool_4_out) #torch.Size([1, 512, 32, 32])
        return (pool_3_out, pool_4_out, pool_5_out)


class TableDecoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(TableDecoder, self).__init__()
        self.conv_7_table = nn.Conv2d(
                        in_channels = 256,
                        out_channels = 256,
                        kernel_size = kernels[0], 
                        stride = strides[0])
        self.upsample_1_table = nn.ConvTranspose2d(
                        in_channels = 256,
                        out_channels=128,
                        kernel_size = kernels[1],
                        stride = strides[1])
        self.upsample_2_table = nn.ConvTranspose2d(
                        in_channels = 128 + channels[0],
                        out_channels = 256,
                        kernel_size = kernels[2],
                        stride = strides[2])
        self.upsample_3_table = nn.ConvTranspose2d(
                        in_channels = 256 + channels[1],
                        out_channels = 1,
                        kernel_size = kernels[3],
                        stride = strides[3])

    def forward(self, x, pool_3_out, pool_4_out):
        x = self.conv_7_table(x)  #[1, 256, 32, 32]
        out = self.upsample_1_table(x) #[1, 128, 64, 64]
        out = torch.cat((out, pool_4_out), dim=1) #[1, 640, 64, 64]
        out = self.upsample_2_table(out) #[1, 256, 128, 128]
        out = torch.cat((out, pool_3_out), dim=1) #[1, 512, 128, 128]
        out = self.upsample_3_table(out) #[1, 1, 1024, 1024]
        return out

class TableNet(nn.Module):
    def __init__(self,encoder = 'vgg', use_pretrained_model = True, basemodel_requires_grad = True):
        super(TableNet, self).__init__()
        
        self.kernels = [(1,1), (2,2), (2,2),(8,8)]
        self.strides = [(1,1), (2,2), (2,2),(8,8)]
        self.in_channels = 512
        
        self.base_model = VGG19(pretrained = use_pretrained_model, requires_grad = basemodel_requires_grad)
        self.pool_channels = [512, 256]

        #common layer
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = self.in_channels, out_channels = 256, kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8))

        self.table_decoder = TableDecoder(self.pool_channels, self.kernels, self.strides)

    def forward(self, x):

        pool_3_out, pool_4_out, pool_5_out = self.base_model(x)
        conv_out = self.conv6(pool_5_out) #[1, 256, 32, 32]
        table_out = self.table_decoder(conv_out, pool_3_out, pool_4_out) #torch.Size([1, 1, 1024, 1024])
        return table_out
#---------------------------------------------------------------------------------------------------------------#

model = U_Net().to(device)
#model = R2U_Net().to(device)
#model = TableNet().to(device)

def dice_loss(pred, target, smooth = 1e-5):
    num = pred.size(0)
    m1 = pred.view(num, -1)  # flatten
    m2 = target.view(num, -1)  # flatten
    intersection = (m1*m2).sum()
    union = m1.sum() + m2.sum()
    
    dice = 2.0 * (intersection + smooth) / (union+ smooth)    
    loss = 1.0 - dice

    acc = dice / (2 - dice)
    
    return loss, acc


num_epochs = 30
batch_size = 2

train_dl, val_dl, train_size, val_size = GetTrainVal(batch_size)


# helper functions
loss_fn = nn.BCELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, min_lr=1e-9)
train_size = train_size
val_size = val_size
epoch_train_loss_list = []
epoch_val_loss_list = []
epoch_val_acc_list = []

print('--------------------------------start--------------------------------------')
for epoch in range(1,num_epochs+1):
    print('---------------------------------------------------------------------------')
    print('epoch %s' % epoch)

    running_loss = []
    val_loss = []
    val_acc = []

    # Training
    model.train()
    for image, truth in train_dl:
        predictions = model(image.cuda())
        loss = dice_loss(predictions, truth.cuda())[0]
        #loss = loss_fn(predictions, truth.cuda())
        loss.requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss.append(loss)
    # Validation
    model.eval()
    with torch.no_grad():
        for image1, truth1 in val_dl:
            predictions = model(image.cuda())
            loss = dice_loss(predictions, truth.cuda())[0]
            #loss = loss_fn(predictions, truth.cuda())
            val_loss.append(loss)
            val_acc.append(dice_loss(predictions, truth.cuda())[1]*100)
    
    epoch_train_loss = sum(running_loss) / len(running_loss)
    epoch_val_loss = sum(val_loss) / len(val_loss)
    epoch_val_acc = sum(val_acc) / len(val_acc)
    scheduler.step(epoch_val_loss) # LR Scheduler

    epoch_train_loss_list.append(epoch_train_loss.cpu().detach().numpy())
    epoch_val_loss_list.append(epoch_val_loss.cpu().detach().numpy())
    epoch_val_acc_list.append(epoch_val_acc.cpu().detach().numpy())


    print(f"==>train_loss: {epoch_train_loss} ==>val_loss: {epoch_val_loss} ==>val_accuracy: {epoch_val_acc}")
    epoch += 1
    if (epoch-1)/10 in range(1, 11):
        
        a = open('Development_DL\Arbeitbreich_DL\model_%S.pkl' %(epoch-1))
        a.close()
        torch.save(model, 'Development_DL\Arbeitbreich_DL\model_%s.pkl' %(epoch-1))

        print('save done')
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



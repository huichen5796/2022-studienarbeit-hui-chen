#https://github.com/pranjalrai-iitd/Fetal-head-segmentation-and-circumference-measurement-from-ultrasound-images/blob/master/Unet.py
#https://www.jianshu.com/p/7086ded792b2
#https://github.com/codecat0/CV/blob/main/Semantic_Segmentation/SegNet/nets/segent.py
#https://medium.com/analytics-vidhya/table-extraction-using-deep-learning-3c91790aa200

import torch
import torch.nn as nn
import os 
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
device = 'cuda'

# make Dataset class
class MakeDataset(Dataset):
    def __init__(self, root_image1, root_mask1, transform = None):
        images_list = [pp for pp in os.listdir(root_image1)]
        masks_list = [pp.replace('.jpg', '_table_mask.jpg') for pp in images_list]
        self.path_images = [os.path.join(root_image1, fn) for fn in images_list]
        self.path_masks = [os.path.join(root_mask1, fn) for fn in masks_list]
        self.transform = transform


    def __len__(self):
        return len(self.path_images)
    
    def __getitem__(self, idx):
        path_image = self.path_images[idx]
        path_mask = self.path_masks[idx]

        image =  torch.FloatTensor(np.array(Image.open(path_image).convert('L'))/255.0).reshape(1,1024,1024)
        mask =  torch.FloatTensor(np.array(Image.open(path_mask).convert('L'))/255.0).reshape(1,1024,1024)
        
        return image, mask


def GetTrainVal(batch_size):

    root_image = "/content/drive/MyDrive/githui/marmot/image"
    root_mask = "/content/drive/MyDrive/githui/marmot/table_mask"

    root_image_val = "/content/drive/MyDrive/githui/marmot/image_val"
    root_mask_val = "/content/drive/MyDrive/githui/marmot/table_mask_val"
    

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
        # torch.Size([ , 1, 1024, 1024]) torch.float32
        print('every batch of train mask: ',mask.shape, mask.dtype)
        # torch.Size([ , 1, 1024, 1024]) torch.float32
        break

    for img, mask in val_dl:
        print('every batch of val image: ',img.shape, img.dtype)
        # torch.Size( , 1, 1024, 1024]) torch.float32
        print('every batch of val mask: ',mask.shape, mask.dtype)
        # torch.Size([ , 1, 1024, 1024]) torch.float32
        break

    return train_dl, val_dl, len(train_ds), len(val_ds)

#GetTrainVal()
'''
The total number of images in the train dataset:  793
The total number of images in the validation dataset:  200
every batch of train image:  torch.Size([, 1, 1024, 1024]) torch.float32
every batch of train mask:  torch.Size([, 1, 1024, 1024]) torch.float32
every batch of val image:  torch.Size([, 1, 1024, 1024]) torch.float32
every batch of val mask:  torch.Size([, 1, 1024, 1024]) torch.float32

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
#---------------------------------------------------------------------------------------------------------------#

model = U_Net().to(device)

def dice_loss(pred, target, smooth = 1e-5):
    num = pred.size(0)
    m1 = pred.view(num, -1)  # flatten
    m2 = target.view(num, -1)  # flatten
    intersection = (m1*m2).sum()
    union = m1.sum() + m2.sum()
    
    dice = 2.0 * (intersection + smooth) / (union+ smooth)    
    dice_loss = 1.0 - dice

    iou = dice / (2 - dice)
    iou_loss = 1-iou
    return dice_loss, iou_loss

bce_loss = nn.BCELoss()

def loss_fn(pred, truth):
    d_loss,i_loss = dice_loss(pred, truth, smooth = 1e-5)
    b_loss = bce_loss(pred, truth)

    loss = b_loss

    return loss


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


num_epochs = 200
batch_size = 2

train_dl, val_dl, train_size, val_size = GetTrainVal(batch_size)

# helper functions

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, min_lr=1e-11)
train_size = train_size
val_size = val_size
epoch_train_loss_list = []
epoch_val_loss_list = []
epoch_val_acc_list = []

print('--------------------------------start--------------------------------------')
for epoch in range(1,num_epochs+1):
    print('---------------------------------------------------------------------------')
    print('epoch %s/%d' % (epoch, num_epochs))

    current_lr = get_lr(optimizer)

    running_loss = []
    val_loss = []
    val_acc = []

    # Training
    model.train()
    for image, truth in train_dl:
        predictions = model(image.cuda())
        loss = loss_fn(predictions, truth.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss.append(loss)
    # Validation
    model.eval()
    with torch.no_grad():
        for image1, truth1 in val_dl:
            predictions = model(image.cuda())

            loss = loss_fn(predictions, truth.cuda())
            val_loss.append(loss)
            val_acc.append((1-dice_loss(predictions, truth.cuda())[1])*100)
    
    epoch_train_loss = sum(running_loss) / len(running_loss)
    epoch_val_loss = sum(val_loss) / len(val_loss)
    epoch_val_acc = sum(val_acc) / len(val_acc)
    scheduler.step(epoch_val_loss) # LR Scheduler

    epoch_train_loss_list.append(epoch_train_loss.cpu().detach().numpy())
    epoch_val_loss_list.append(epoch_val_loss.cpu().detach().numpy())
    epoch_val_acc_list.append(epoch_val_acc.cpu().detach().numpy())


    print(f"=>train_loss: {epoch_train_loss} =>val_loss: {epoch_val_loss} =>iou_acc: {epoch_val_acc} =>learn-rate: {current_lr}")
    epoch += 1
    if (epoch-1)/10 in range(1, 21): # every 10 epochs save once
        
        torch.save(model, '/content/drive/MyDrive/unet_model_c1_%s.pkl' %(epoch-1))
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



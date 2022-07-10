import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
import cv2
from PIL import Image
import os
import copy
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
import torchvision.models as models
device = 'cuda'

transform = A.Compose([

    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255,
    ),
    ToTensorV2()
])

# make Dataset class


class MakeDataset(Dataset):
    def __init__(self, root_image1, root_mask1):
        images_list = [pp for pp in os.listdir(root_image1)]
        masks_list = [pp.replace('.jpg', '_table_mask.jpg')
                      for pp in images_list]
        self.path_images = [os.path.join(root_image1, fn)
                            for fn in images_list]
        self.path_masks = [os.path.join(root_mask1, fn) for fn in masks_list]

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, idx):
        path_image = self.path_images[idx]

        #image =  torch.FloatTensor(np.array(Image.open(path_image))/255.0).reshape(3,1024,1024)
        img = np.array(Image.open(path_image))
        image = transform(image=img)["image"]

        path_mask = self.path_masks[idx]
        mask = torch.FloatTensor(
            np.array(Image.open(path_mask).convert('L'))/255.0).reshape(1, 1024, 1024)

        return image, mask


def GetTrainVal(batch_size):

    root_image = "/content/drive/MyDrive/githui/marmot/image"
    root_mask = "/content/drive/MyDrive/githui/marmot/table_mask"

    root_image_val = "/content/drive/MyDrive/githui/marmot/image_val"
    root_mask_val = "/content/drive/MyDrive/githui/marmot/table_mask_val"

    # instantiate dataset
    train_ds = MakeDataset(root_image, root_mask)
    val_ds = MakeDataset(root_image_val, root_mask_val)

    print('The total number of images in the train dataset: ', len(train_ds))
    print('The total number of images in the validation dataset: ', len(val_ds))

    # make dataloader
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=True)

    # see the infos of datas
    for img, mask in train_dl:
        print('every batch of train image: ', img.shape, img.dtype)
        # torch.Size([ , 1, 1024, 1024]) torch.float32
        print('every batch of train mask: ', mask.shape, mask.dtype)
        # torch.Size([ , 1, 1024, 1024]) torch.float32
        break

    for img, mask in val_dl:
        print('every batch of val image: ', img.shape, img.dtype)
        # torch.Size( , 1, 1024, 1024]) torch.float32
        print('every batch of val mask: ', mask.shape, mask.dtype)
        # torch.Size([ , 1, 1024, 1024]) torch.float32
        break

    return train_dl, val_dl, len(train_ds), len(val_ds)


# GetTrainVal()

#---------------------------------------------------------------------------------------------------------------#
# model DenseNet

class DenseNet(nn.Module):
    def __init__(self, pretrained=True, requires_grad=True):
        super(DenseNet, self).__init__()
        denseNet = torchvision.models.densenet121(pretrained=True).features
        self.densenet_out_1 = torch.nn.Sequential()
        self.densenet_out_2 = torch.nn.Sequential()
        self.densenet_out_3 = torch.nn.Sequential()

        for x in range(8):
            self.densenet_out_1.add_module(str(x), denseNet[x])
        for x in range(8, 10):
            self.densenet_out_2.add_module(str(x), denseNet[x])

        self.densenet_out_3.add_module(str(10), denseNet[10])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):

        out_1 = self.densenet_out_1(x)  # torch.Size([1, 256, 64, 64])
        out_2 = self.densenet_out_2(out_1)  # torch.Size([1, 512, 32, 32])
        out_3 = self.densenet_out_3(out_2)  # torch.Size([1, 1024, 32, 32])
        return out_1, out_2, out_3


class TableDecoder(nn.Module):
    def __init__(self, channels, kernels, strides):
        super(TableDecoder, self).__init__()
        self.conv_7_table = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=kernels[0],
            stride=strides[0])
        self.upsample_1_table = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=kernels[1],
            stride=strides[1])
        self.upsample_2_table = nn.ConvTranspose2d(
            in_channels=128 + channels[0],
            out_channels=256,
            kernel_size=kernels[2],
            stride=strides[2])
        self.upsample_3_table = nn.ConvTranspose2d(
            in_channels=256 + channels[1],
            out_channels=1,
            kernel_size=kernels[3],
            stride=strides[3])

    def forward(self, x, pool_3_out, pool_4_out):
        x = self.conv_7_table(x)  # [1, 256, 32, 32]
        out = self.upsample_1_table(x)  # [1, 128, 64, 64]
        out = torch.cat((out, pool_4_out), dim=1)  # [1, 640, 64, 64]
        out = self.upsample_2_table(out)  # [1, 256, 128, 128]
        out = torch.cat((out, pool_3_out), dim=1)  # [1, 512, 128, 128]
        out = self.upsample_3_table(out)  # [1, 3, 1024, 1024]
        return out


class TableNet(nn.Module):
    def __init__(self, encoder='densenet', use_pretrained_model=True, basemodel_requires_grad=True):
        super(TableNet, self).__init__()

        self.base_model = DenseNet(
            pretrained=use_pretrained_model, requires_grad=basemodel_requires_grad)
        self.pool_channels = [512, 256]
        self.in_channels = 1024
        self.kernels = [(1, 1), (1, 1), (2, 2), (16, 16)]
        self.strides = [(1, 1), (1, 1), (2, 2), (16, 16)]

        # common layer
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=256, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8))

        self.table_decoder = TableDecoder(
            self.pool_channels, self.kernels, self.strides)

    def forward(self, x):

        pool_3_out, pool_4_out, pool_5_out = self.base_model(x)
        conv_out = self.conv6(pool_5_out)  # [1, 256, 32, 32]
        # torch.Size([1, 1, 1024, 1024])
        table_out = self.table_decoder(conv_out, pool_3_out, pool_4_out)
        return table_out

#---------------------------------------------------------------------------------------------------------------#


model = TableNet().to(device)
#model = torch.load('/content/drive/MyDrive/tablemodel2_100.pkl')


def acc(pred, target, smooth=1e-5):
    s = nn.Sigmoid()
    pred = s(pred)
    num = pred.size(0)
    m1 = pred.view(num, -1)  # flatten
    m2 = target.view(num, -1)  # flatten
    intersection = (m1*m2).sum()
    union = m1.sum() + m2.sum()

    dice = 2.0 * (intersection + smooth) / (union + smooth)

    iou = dice / (2 - dice)

    return dice, iou


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


num_epochs = 100
batch_size = 4

train_dl, val_dl, train_size, val_size = GetTrainVal(batch_size)


# helper functions
bce_loss = nn.BCEWithLogitsLoss()  # sigmoid + BCE
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=3e-4)
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=4, min_lr=1e-12)
train_size = train_size
val_size = val_size


epoch_train_loss_list = []
epoch_val_loss_list = []
epoch_val_acc_list = []
accumulation_steps = 8  # gradient accumulation

print('--------------------------------start--------------------------------------')
for epoch in range(1, num_epochs+1):
    print('---------------------------------------------------------------------------')
    print('epoch %s' % epoch)

    current_lr = get_lr(optimizer)

    running_loss = []
    val_loss = []
    val_acc = []

    # Training
    model.train()
    for i, (image, truth) in enumerate(train_dl):
        predictions = model(image.cuda())

        loss = bce_loss(predictions, truth.cuda())
        running_loss.append(loss)

        loss = loss/accumulation_steps

        loss.backward()
        if (i+1) % accumulation_steps == 0:

            optimizer.step()
            optimizer.zero_grad()

    # Validation

    model.eval()
    with torch.no_grad():
        for image1, truth1 in val_dl:
            predictions = model(image.cuda())

            loss = bce_loss(predictions, truth.cuda())
            val_loss.append(loss)
            val_acc.append(acc(predictions, truth.cuda())[0]*100)

    epoch_train_loss = sum(running_loss) / len(running_loss)
    epoch_val_loss = sum(val_loss) / len(val_loss)
    epoch_val_acc = sum(val_acc) / len(val_acc)
    scheduler.step(epoch_val_loss)  # LR Scheduler

    epoch_train_loss_list.append(epoch_train_loss.cpu().detach().numpy())
    epoch_val_loss_list.append(epoch_val_loss.cpu().detach().numpy())
    epoch_val_acc_list.append(epoch_val_acc.cpu().detach().numpy())

    print(
        f"==>train_loss: {epoch_train_loss} ==>val_loss: {epoch_val_loss} ==>dice_acc: {epoch_val_acc} ==>learn-rate: {current_lr} ")

    epoch += 1
    if (epoch-1)/10 in range(1, 21):

        torch.save(model, '/content/drive/MyDrive/tablemodel4_%s.pkl' %
                   (epoch-1))
        print('save done')
# Visualize the results

plt.title("Train-Val Loss")
plt.plot(range(1, num_epochs+1), epoch_train_loss_list, label="train")
plt.plot(range(1, num_epochs+1), epoch_val_loss_list, label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()


# plot accuracy progress
plt.title("Val Accuracy")
plt.plot(range(1, num_epochs+1), epoch_val_acc_list)
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

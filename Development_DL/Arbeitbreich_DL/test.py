import torch
import torch.nn.functional 
from torchvision.transforms.functional import to_tensor
import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn



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

        self.Conv1 = conv_block(img_ch,16)
        self.Conv2 = conv_block(16,32)
        self.Conv3 = conv_block(32,64)
        self.Conv4 = conv_block(64,128)
        self.Conv5 = conv_block(128,256)

        self.Up5 = up_conv(256,128)
        self.Up_conv5 = conv_block(256, 128)

        self.Up4 = up_conv(128,64)
        self.Up_conv4 = conv_block(128,64)
        
        self.Up3 = up_conv(64,32)
        self.Up_conv3 = conv_block(64,32)
        
        self.Up2 = up_conv(32,16)
        self.Up_conv2 = conv_block(32,16)

        self.Conv_1x1 = nn.Sequential(nn.Conv2d(16,output_ch,kernel_size=1,stride=1,padding=0),
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




model = torch.load('/content/sample_data/model.pkl')
root_image = "/content/drive/MyDrive/githui/2022-projektarbeit-hui-chen/Development_DL/Arbeitbreich_DL/marmot/image"
root_mask = "/content/drive/MyDrive/githui/2022-projektarbeit-hui-chen/Development_DL/Arbeitbreich_DL/marmot/table_mask"

np.random.seed(1)
imgs_list = [pp for pp in os.listdir(root_image)]
rnd_imgs = np.random.choice(imgs_list, 4)
masks_list = [pp for pp in os.listdir(root_mask)]


i = 0
plt.figure(figsize=(10, 10))
for fn in rnd_imgs:
        
    img_path = os.path.join(root_image, fn)
    mask_path = os.path.join(root_mask, fn.replace('.jpg', '_table_mask.jpg'))
        
    img = cv2.imread(img_path, 0)
    image = cv2.resize(img, (1024, 1024))
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = image.astype(np.float32)
    image /= 255.0 # Normalisierung

    mask = cv2.imread(mask_path, 0)

    image = image.astype('uint8')    
    image = to_tensor(image)
    image = image.unsqueeze(0)
    
    pred = model(image.cuda())

    pred = torch.squeeze(pred, dim=0)
    pred = torch.squeeze(pred, dim=0)
    pred = np.array(pred.cpu().detach().numpy())
    print(pred)
    #pred = pred.astype(np.uint8)
    pred[pred > 0.5] = 255



    plt.subplot(4, 3, 3*i+1) 
    plt.imshow(img, cmap="gray")

    plt.subplot(4, 3, 3*i+2) 
    plt.imshow(mask, cmap="gray")

    plt.subplot(4, 3, 3*i+3) 
    plt.imshow(pred, cmap='gray' )
    i+=1

plt.show()
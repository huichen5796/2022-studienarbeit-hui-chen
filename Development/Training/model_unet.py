import torch
import torch.nn as nn

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
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2))

    def forward(self,x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, 32)
        self.Conv2 = conv_block(32, 64)
        self.Conv3 = conv_block(64, 128)

        self.Conv4 = conv_block(128, 256)
        self.Conv5 = conv_block(256, 512)

        self.Up5 = up_conv(512, 256)
        self.Up_conv5 = conv_block(512, 256)

        self.Up4 = up_conv(256, 128)
        self.Up_conv4 = conv_block(256, 128)

        self.Up3 = up_conv(128, 64)
        self.Up_conv3 = conv_block(128, 64)

        self.Up2 = up_conv(64, 32)
        self.Up_conv2 = conv_block(64, 32)

        self.Conv_1x1 = nn.Sequential(nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0),
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

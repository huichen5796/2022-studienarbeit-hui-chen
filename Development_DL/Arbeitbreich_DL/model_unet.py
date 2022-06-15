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

# CPU or GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

        
        path_mask = self.path_masks[idx]
        mask = cv2.imread(path_mask,0)
        mask = cv2.resize(mask, (1024, 1024))


        
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
class UNet(nn.Module):
    def __init__(self, params):
        super(UNet, self).__init__()
        C_in, H_in, W_in = params['input_shape'] # C_in is cannel of input image
        init_f = params['initial_filters']
        num_outputs = params['num_outputs']
        # NN
        self.conv1 = nn.Conv2d(C_in, init_f, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(8*init_f, 16*init_f, kernel_size=3, stride=1, padding=1)
        # Define the upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_up1 = nn.Conv2d(16*init_f, 8*init_f, kernel_size=3, stride=1, padding=1)
        self.conv_up2 = nn.Conv2d(8*init_f, 4*init_f, kernel_size=3, stride=1, padding=1)
        self.conv_up3 = nn.Conv2d(4*init_f, 2*init_f, kernel_size=3, stride=1, padding=1)
        self.conv_up4 = nn.Conv2d(2*init_f, init_f, kernel_size=3, stride=1, padding=1)
        
        self.conv_out = nn.Conv2d(init_f, num_outputs, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv5(x))
        
        x = self.upsample(x)
        x = F.relu(self.conv_up1(x))

        x = self.upsample(x)
        x = F.relu(self.conv_up2(x))
        
        x = self.upsample(x)
        x = F.relu(self.conv_up3(x))
        
        x = self.upsample(x)
        x = F.relu(self.conv_up4(x))

        x = self.conv_out(x)
        
        return x
        
params_model={
        "input_shape": (1, 1024, 1024),
        "initial_filters": 16, 
        "num_outputs": 1,
            }

model = UNet(params_model).to(device)

print(model)

# """
# UNet(
#  (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))      
#  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     
#  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     
#  (conv4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))    
#  (conv5): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))   
#  (upsample): Upsample(scale_factor=2.0, mode=bilinear)
#  (conv_up1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#  (conv_up2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
#  (conv_up3): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
#  (conv_up4): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
#  (conv_out): Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))   
#  )
# """


## define loss function
# Dice coefficient is a set similarity measure function, 
# which is usually used to calculate the similarity between two samples, and its value range is [0,1]
# https://zhuanlan.zhihu.com/p/86704421
def dice_loss(pred, target, smooth = 1e-5):
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) 
    
    dice = 2.0 * (intersection + smooth) / (union+ smooth)    
    loss = 1.0 - dice
    
    return loss.sum(), dice.sum()

def loss_func(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target,  reduction='sum')
    
    pred = torch.sigmoid(pred)
    dlv, _ = dice_loss(pred, target)
    
    loss = bce  + dlv

    return loss

## helper functions 
# get learn rate
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# Define an evaluation function
def metrics_batch(pred, target):
    pred = torch.sigmoid(pred)
    _, metric = dice_loss(pred, target)
    
    return metric

# Loss calculation for each batch
def loss_batch(loss_func, output, target, opt=None):   
    loss = loss_func(output, target)
    
    with torch.no_grad():
        pred = torch.sigmoid(output)
        _, metric_b = dice_loss(pred, target)
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b

# Calculations for each round
def loss_epoch(model,loss_func,dataset_dl,sanity_check=False,opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        
        output = model(xb)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        running_loss += loss_b
        
        if metric_b is not None:
            running_metric += metric_b

        if sanity_check is True:
            break
    
    loss = running_loss / float(len_data)
    
    metric = running_metric / float(len_data)
    
    return loss, metric


# train main function
def train_val(model, params):
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    sanity_check = params["sanity_check"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]
    
    loss_history = {
        "train": [],
        "val": []}
    
    metric_history = {
        "train": [],
        "val": []}    
    
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')    
    
    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))   

        model.train()
        train_loss, train_metric = loss_epoch(model,loss_func,train_dl,sanity_check,opt)

        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model,loss_func,val_dl,sanity_check)
       
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)   
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")
            
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts) 
            
        print("train loss: %.6f, accuracy: %.2f" %(train_loss, 100*train_metric))
        print("val loss: %.6f, accuracy: %.2f" %(val_loss, 100*val_metric))
        print("-"*10) 
        

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history


# define optimizer function and how to new the parameter
opt = optim.Adam(model.parameters(), lr=3e-4)
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)

path_models = "./models/"
if not os.path.exists(path_models):
    os.mkdir(path_models)

train_dl, val_dl = GetTrainVal()

params_train={
    "num_epochs": 10,
    "optimizer": opt,
    "loss_func": loss_func,
    "train_dl": train_dl,
    "val_dl": val_dl,
    "sanity_check": False,
    "lr_scheduler": lr_scheduler,
    "path2weights": path_models+"weights.pt",
}

model, loss_hist, metric_hist = train_val(model,params_train)



## Visualize the results
num_epochs=params_train["num_epochs"]

plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

# plot accuracy progress
plt.title("Train-Val Accuracy")
plt.plot(range(1,num_epochs+1),metric_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),metric_hist["val"],label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

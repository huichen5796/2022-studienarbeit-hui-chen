#https://github.com/pranjalrai-iitd/Fetal-head-segmentation-and-circumference-measurement-from-ultrasound-images/blob/master/Unet.py
#https://www.jianshu.com/p/7086ded792b2

import torch
import torch.nn as nn
import torch.nn.functional as F
from make_val_train_dataset import GetTrainVal
import os 
import copy
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# CPU or GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# encoder-decoder model  U-Net
class SegNet(nn.Module):
    def __init__(self, params):
        super(SegNet, self).__init__()
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
        "input_shape": (3, 1024, 1024),
        "initial_filters": 128, 
        "num_outputs": 3,
            }

model = SegNet(params_model).to(device)

#print(model)

# """
# SegNet(
#  (conv1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#  (conv2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#  (conv3): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#  (conv4): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#  (conv5): Conv2d(1024, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#  (upsample): Upsample(scale_factor=2.0, mode=bilinear)
#  (conv_up1): Conv2d(2048, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#  (conv_up2): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#  (conv_up3): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#  (conv_up4): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#  (conv_out): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# )
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

path_models = "./models/sos/"
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

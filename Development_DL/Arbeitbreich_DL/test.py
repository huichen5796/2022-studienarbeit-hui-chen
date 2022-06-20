import torch
import torch.nn.functional 
from torchvision.transforms.functional import to_tensor
import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt



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


    plt.subplot(4, 3, 3*i+1) 
    plt.imshow(img, cmap="gray")

    plt.subplot(4, 3, 3*i+2) 
    plt.imshow(mask, cmap="gray")

    plt.subplot(4, 3, 3*i+3) 
    plt.imshow(pred.cpu().detach().numpy(), cmap='gray' )
    i+=1

plt.show()
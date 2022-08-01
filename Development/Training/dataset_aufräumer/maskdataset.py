import cv2
import os
import numpy as np
from PIL import Image

def GetImageList(dir_name):
    '''
    get all the filename of images unter a dir

    - input: path of dir

    - output: all the files unter the dir, in form list 

    '''

    try:
        file_list = os.listdir(dir_name)
        # if the input is a path of dir then get all the name of the files unter the dir

        return file_list

    except Exception as e:
        print('ERROR BY GetImageList: ' + str(e))


def MakeMask(file_list):
    for file in file_list:
        if 'k' is not os.path.splitext(os.path.basename(file))[0][-1]:
            img = cv2.imread('dataset\label' + '\\' + file)
            cv2.imwrite('dataset\labels' + '\\' + os.path.splitext(os.path.basename(file))[0] + '_mask.png', np.zeros((img.shape[0], img.shape[1],3)))
    

def SizeNormalize(img):
    '''
    Normalize the input image size to 1024 x 1024

    - input: image, 1 channel or 3 channel

    - output: image 1024 x 1024

    '''
    # at first normalize the shape to 1024 X () if size > 1024
    shape_list = list(img.shape)

    if max(shape_list) > 1024:
        scaling_r = 1024/max(shape_list)
        # w, h to avoid exceeding 1024, subtract one
        shape_new = [int(shape_list[1]*scaling_r-1),
                     int(shape_list[0]*scaling_r-1)]
        shape_new[shape_new.index(max(shape_new))] = int(1024)
        image = cv2.resize(img, shape_new)

        if __name__ == '__main__':
            print('image_shape ==> ' + str(img.shape) +
                  ' new ==> ' + str(image.shape))
    else:
        image = img
    # The model can only process pictures of 1024x1024 size,
    # so it is necessary to fill the edges of pictures smaller than this size

    h = image.shape[0]
    w = image.shape[1]
    top = (1024-h)//2
    bottom = 1024-h-top
    left = (1024-w)//2
    right = 1024-w-left
    img_1024 = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    img_1024 = np.array(Image.fromarray(
        cv2.cvtColor(img_1024, cv2.COLOR_BGR2RGB)))

    return img_1024

file_list = GetImageList('dataset\labels')
print(file_list)

for file in file_list:

    img = cv2.imread('dataset\labels' + '\\' + file, 0)
    img_1024 = SizeNormalize(img)
    cv2.imwrite('dataset\\nordataset\label' + '\\' + file, img_1024)
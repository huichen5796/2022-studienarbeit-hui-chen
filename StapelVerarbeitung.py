
import os
from dl_TableExtrakt import Main

def GetImageList(dir_name): 
 
    # get all the filename of images unter a dir
    
    file_list = os.listdir(dir_name) # if the input is a dir then get all the name of the files unter the dir
    
    return file_list
    

def StapelVerbreitung(dir):

    image_list = GetImageList(dir)
    print('---------------------------')
    print('There are %s images in total.'%len(image_list))
    print('---------------------------')
    # print(image_list)
    path_images = [os.path.normpath(os.path.join(dir, fn)) for fn in image_list]
    for image in path_images:
        Main(image)

    
if __name__ == '__main__':
    StapelVerbreitung('Development_tradionell\imageTest')
    
import os
import datetime
import cv2
import numpy

class SquareNormaler:
    def __init__(self, image, size = 1024, color = (255, 255, 255)):
        self.image = image
        self.size = size
        self.color = color

    def size_normaler(self):
        shape_list = list(self.image.shape) # [h, w, c]

        if max(shape_list) > self.size:
            scaling_r = self.size/max(shape_list)

            shape_new = [int(shape_list[1]*scaling_r-1),
                        int(shape_list[0]*scaling_r-1)]
            shape_new[shape_new.index(max(shape_new))] = int(self.size)
            image = cv2.resize(self.image, shape_new)

        else:
            image = self.image

        h = image.shape[0]
        w = image.shape[1]
        top = (self.size-h)//2
        bottom = self.size-h-top
        left = (self.size-w)//2
        right = self.size-w-left
        square_image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)
        
        square_image = cv2.cvtColor(square_image, cv2.COLOR_BGR2RGB)

        return square_image
import os
import datetime
import cv2
import numpy

DEFAULT_BORDER_SIZE = 50

class ImageUntils:
    def __init__(self, image, boundRects):
        self.image = image
        self.image_hwc = image.shape
        self.boundRects = boundRects
        
    def image_highlight(self):
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
        i = 0
        color_image = numpy.ones((self.image_hwc[0], self.image_hwc[1], 3), numpy.uint8)*255
        for x, y, w, h in self.boundRects:
            size = 0
            triangle = numpy.array(
                [[x-size, y-size], [x-size, y+h+size], [x+w+size, y+h+size], [x+w+size, y-size]])
            cv2.fillConvexPoly(color_image, triangle, color[i])
            i += 1
            if i > 3:
                i = 0
        
        if len(list(self.image_hwc)) == 2:
            image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

        return cv2.addWeighted(image, 0.9, color_image, 0.5, 0)
    
    def image_cut(self, border_size = DEFAULT_BORDER_SIZE, border_color = [255,255,255]):
        cutted_zones = [None]*len(self.boundRects)
        for ii, (x, y, w, h) in enumerate(self.boundRects):
            t = border_size
            cutted_zones[ii] = numpy.ones((h, w, len(border_color)))
            cutted_zones[ii] = cv2.copyMakeBorder(self.image[(y):(
                y+h), (x):(x+w)], t, t, t, t, cv2.BORDER_CONSTANT, value=border_color)

        return cutted_zones
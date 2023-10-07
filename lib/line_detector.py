import os
import datetime
import cv2
import numpy

class LineDetector:
    def __init__(self, image, minLong):
        self.image = self.pre_processing(image)
        self.minLong = minLong

    def pre_processing(self, image):
        image = image if len(list(image.shape)) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 15)
        return image

    def fld(self):
        fld = cv2.ximgproc.createFastLineDetector()
        dlines = fld.detect(self.image)
        longLines = []
        if dlines is not None:
            for dline in dlines:
                x0, y0, x1, y1 = [int(round(coord)) for coord in dline[0]]
                long = (y1-y0)*(y1-y0)+(x1-x0)*(x1-x0)
                if long >= self.minLong*self.minLong:
                    longLines.append([x0, y0, x1, y1])

        return longLines
            
    def lsd(self):
        lsd = cv2.createLineSegmentDetector(0, scale=1)
        dlines = lsd.detect(self.image)
        longLines = []
        if dlines is not None:
            for dline in dlines[0]:
                x0, y0, x1, y1 = [int(round(coord)) for coord in dline[0]]
                long = (y1-y0)*(y1-y0)+(x1-x0)*(x1-x0)
                if long >= self.minLong*self.minLong:
                    longLines.append([x0, y0, x1, y1])

        return longLines
    
    def make_canvas(self):
        return numpy.zeros((self.image.shape[0], self.image.shape[1]))
    
    def show(self, detector):
        canvas = self.make_canvas()
        if detector == 'fld':
            lines = self.fld()
        elif detector == 'lsd':
            lines = self.lsd()
        for [x0, y0, x1, y1] in lines:
            cv2.line(canvas, (x0, y0), (x1, y1), color=255, thickness=4, lineType=cv2.LINE_AA)
        
        return canvas
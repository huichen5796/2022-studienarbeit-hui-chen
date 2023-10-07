import os
import datetime
import cv2
import numpy
import math

from .line_detector import LineDetector

class TiltCorrector:
    def __init__(self, image):
        self.image = image
        self.longLines = LineDetector(image, (image.shape[0]+image.shape[1])//4).fld()

    def get_line_angle(self):
        if len(self.longLines) == 0:
            raise ValueError("no lines detected")
        else:
            angles = []
            for line in self.longLines:
                x0, y0, x1, y1 = line
                if x0 == x1:
                    angles.append(90)
                elif y0 == y1:
                    angles.append(0)
                else:
                    t = float(y1-y0)/(x1-x0)
                    rotate_angle = (math.degrees(math.atan(t)))
                    angles.append(rotate_angle)
            angles_m45 = [angles[i] for i in range(len(angles)) if abs(angles[i]) < 45]
            if len(angles_m45) == 0:
                angle_average = 0
            else:
                angle_average = (sum(angles_m45)/len(angles_m45))

            return angle_average
    
    def get_box_angle(self):
        bina_image = cv2.adaptiveThreshold(
            self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 10)
        # Gaussian binarization, invert, noise reduction by opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        bina_image = cv2.bitwise_not(bina_image)
        bina_image = cv2.morphologyEx(bina_image, cv2.MORPH_OPEN, kernel)

        coords = numpy.where(bina_image > 0)

        points = [None]*len(coords[0])
        for i, x in enumerate(coords[0]):
            y = coords[1][i]
            points[i] = (y, x)
        points = numpy.array(points)
        rect = cv2.minAreaRect(points)
        angle = (rect[2])

        return angle if angle <=45 else (angle-90)
    
    def rotate(self):
        try:
            angle = self.get_line_angle()
        except:
            angle = self.get_box_angle()

        h, w = self.image.shape[0:2]
        center_X, center_Y = w // 2, h // 2
        M = cv2.getRotationMatrix2D((center_X, center_Y), angle, 1.0)
        cos = numpy.abs(M[0, 0])
        sin = numpy.abs(M[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w / 2) - center_X
        M[1, 2] += (new_h / 2) - center_Y

        image_rotate = cv2.warpAffine(
            self.image, M, (new_w, new_h), borderValue=(255, 255, 255))
        
        if abs(angle) > 25:
            image_rotate = remove_border(image_rotate)

        return image_rotate
    

def remove_border(image):
    image = cv2.copyMakeBorder(
        image, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
    bina_image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 10)

    x, y, w, h = cv2.boundingRect(cv2.bitwise_not(bina_image))
    thickness = 25

    text_zone = numpy.ones((h+2*thickness, w+2*thickness, 1))
    text_zone = image[(y-thickness):(y+h+thickness),
                            (x-thickness):(x+w+thickness)]

    return text_zone
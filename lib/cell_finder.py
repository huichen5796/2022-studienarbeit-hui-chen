import os
import datetime
import cv2
import numpy

from .image_untils import ImageUntils
from .line_detector import LineDetector

class CellFinder:
    def __init__(self, square_image, table_boundRects):
        self.square_image = square_image
        self.table_boundRects = table_boundRects

    def line_deleter(self, image):
        image_threshold = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 15)
        image_threshold = numpy.array(image_threshold, numpy.uint8)
        
        canvas_lines = LineDetector(image, 18).show('fld')
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        canvas_lines = numpy.array(canvas_lines, numpy.uint8)
        canvas_lines = cv2.dilate(canvas_lines, kernel, iterations=1)
        canvas_lines = cv2.erode(canvas_lines, kernel, iterations=1)
        canvas_lines = numpy.array(canvas_lines, numpy.uint8)

        image_threshold_lineDeleted = cv2.bitwise_or(image_threshold, canvas_lines)

        return image_threshold_lineDeleted
    
    def cell_finder(self, image_threshold_lineDeleted):
        image_threshold_lineDeleted_inv = cv2.bitwise_not(image_threshold_lineDeleted)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 3))
        bina_image = cv2.dilate(image_threshold_lineDeleted_inv, kernel, iterations=1)
        bina_image = cv2.erode(bina_image, kernel, iterations=1)

        _, bina_image = cv2.threshold(
            bina_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            bina_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        mask_image = numpy.zeros((image_threshold_lineDeleted_inv.shape[0], image_threshold_lineDeleted_inv.shape[1], 1), numpy.uint8)
        size = 2
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 4 and h > 5 and w < 0.5*image_threshold_lineDeleted_inv.shape[1]:
                x = int(x - size)
                y = int(y)
                w = int(w + 2 * size)
                h = int(h)

                # cv2.rectangle(color_image, (x, y), (x+w, y+h), 0, 2)  # round the text zone by rect
                triangle = numpy.array([[x, y], [x, y+h], [x+w, y+h], [x+w, y]])
                cv2.fillConvexPoly(mask_image, triangle, 255)
        contours, _ = cv2.findContours(
            mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cell_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 0.7*image_threshold_lineDeleted_inv.shape[1]:
                x = int(x)
                y = int(y - size)
                w = int(w)
                h = int(h + 2 * size)
                cell_contours.append((x, y, w, h))
        
        return cell_contours
    
    def run(self):
        task = ImageUntils(self.square_image, self.table_boundRects)
        table_zones = task.image_cut()
        cell_boundRects_relativ = []
        for table_zone in table_zones:
            table_zone = cv2.cvtColor(table_zone, cv2.COLOR_BGR2GRAY)
            image_threshold_lineDeleted = self.line_deleter(table_zone)
            cell_boundRects_relativ.append(self.cell_finder(image_threshold_lineDeleted))
        
        return cell_boundRects_relativ


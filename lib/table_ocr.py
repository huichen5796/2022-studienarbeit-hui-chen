import os
import datetime
import cv2
import numpy
from .image_untils import ImageUntils
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'D:/for_tesseract/tesseract.exe'

class TableOcr:
    def __init__(self, square_image, table_boundRects, cell_boundRects_relativ_withLabel):
        self.square_image = square_image
        self.table_boundRects = table_boundRects
        self.cell_boundRects_relativ_withLabel = cell_boundRects_relativ_withLabel

    def cell_ocr(self, cell_image):
        result = pytesseract.image_to_string(
        cell_image, lang='deu', config='--psm 7')
        if '\n' in result:
            result = result.replace('\n', '').replace('|', '').replace('/', '')
        if result == '':
            result = '(unknown)'

        return result

    def table_ocr(self, table_zone, cells):
        size = 0
        for cell in cells[0]:
            row_label, col_lable, w, h, x, y = cell
            cell_zone = numpy.ones((h+2*size, w+2*size, 1))
            cell_zone = table_zone[(y-size):(y+h+size), (x-size):(x+w+size)]
            cell_zone = cv2.resize(
                cell_zone, (cell_zone.shape[1]*4, cell_zone.shape[0]*4))

            value = [None, None, None]
            for cha in range(3):
                value[cha] = (numpy.mean(cell_zone[cha], axis=0)[0]
                            + numpy.mean(cell_zone[cha], axis=0)[-1]
                            + numpy.mean(cell_zone[cha], axis=1)[0]
                            + numpy.mean(cell_zone[cha], axis=1)[-1]) // 4

            cell_zone = cv2.copyMakeBorder(
                cell_zone, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=value)

            result = self.cell_ocr(cell_zone)
            cell.append(result)

        return cells
    
    def run(self):
        cells_infos = []
        task_object = ImageUntils(self.square_image, self.table_boundRects)
        cutted_zones = task_object.image_cut(border_color=[0])
        for i in range(0, len(cutted_zones)):
            cells_infos.append(self.table_ocr(cutted_zones[i], self.cell_boundRects_relativ_withLabel[i]))

        return cells_infos
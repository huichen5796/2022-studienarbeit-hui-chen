import os
import datetime
import cv2
import numpy

class CellLabelGenerator:
    def __init__(self, column_boundRects_relativ, cell_boundRects_relativ):
        self.column_boundRects_relativ = column_boundRects_relativ
        self.cell_boundRects_relativ = cell_boundRects_relativ
        if len(column_boundRects_relativ) != len(cell_boundRects_relativ):
            raise ValueError('cell label generator error')

    def corrector(self, column_boundRect, cell_boundRect):
        cell_boundRect = sorted(cell_boundRect, key=lambda x: x[0])
        processed = []
        for i, cell in enumerate(cell_boundRect):    
            for col, w in column_boundRect:
                if abs(cell[0] - col) <= w//2:
                    cell[0] = col
                    processed.append(i)
                    break
        for i, cell in enumerate(cell_boundRect):
            if i not in processed:
                if i == 0:
                    cell[0] = cell_boundRect[1][0]
                else:
                    cell[0] = cell_boundRect[i-1][0]
        cell_boundRect = sorted(cell_boundRect, key=lambda x: (x[1], x[0]))
        for i in range(len(cell_boundRect)-1):
            if cell_boundRect[i+1][1] == cell_boundRect[i][1]:
                continue
            else:
                if abs(cell_boundRect[i+1][1]-cell_boundRect[i][1]) < int(cell_boundRect[i+1][3]*0.3 + cell_boundRect[i][3]*0.7 - 2):
                    cell_boundRect[i+1][1] = cell_boundRect[i][1]
                else:
                    continue
        cell_boundRect = sorted(cell_boundRect, key=lambda x: (x[1], x[0]))

        return cell_boundRect
    
    def run(self):
        cell_boundRects_relativ_withLabel = []
        for i in range(0,len(self.column_boundRects_relativ)):
            center_list = [None]*len(self.cell_boundRects_relativ[i])
            for iii, (x, y, w, h) in enumerate(self.cell_boundRects_relativ[i]):
                center_list[iii] = [x+w//2, y+h//2, w, h, x, y]

            center_list = self.corrector(self.column_boundRects_relativ[i], center_list)
            cols_list = list(set([pp[0] for pp in center_list]))  # alle x axis
            cols_list.sort()
            rows_list = list(set([pp[1] for pp in center_list]))  # alle y axis
            rows_list.sort()
            tablesize = [len(rows_list), len(cols_list)]

            for i, (c_x, c_y, w, h, x, y) in enumerate(center_list):
                center_list[i][0] = int(rows_list.index(c_y))
                center_list[i][1] = f'col{cols_list.index(c_x)}'

            cell_boundRects_relativ_withLabel.append([center_list, tablesize])

        return cell_boundRects_relativ_withLabel
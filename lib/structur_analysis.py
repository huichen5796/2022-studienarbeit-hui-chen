import os
import datetime
import cv2
import numpy
import pandas

from .image_untils import ImageUntils

class SturcturAnalysis:
    def __init__(self,cells_infos):
        self.cells_infos = cells_infos

    def dataframe_generator(self, cells_info):
        keys = ['col%s' % s for s in range(cells_info[1][1])]

        values = [None]*len(keys)
        for i, key in enumerate(keys):
            col_info = []
            index = []
            for m in range(len(cells_info[0])):
                if key in cells_info[0][m]:
                    col_info.append(cells_info[0][m][-1])
                    index.append(cells_info[0][m][0])

            values[i] = pandas.Series(col_info, index=index)
            values[i] = values[i].to_dict()
            values[i] = pandas.Series(values[i])

        dict_info = dict(zip(keys, values))
        df = pandas.DataFrame(dict_info)
        df = df.fillna('(empty_cell)')
        
        return df

    def run(self):
        for cells_info in self.cells_infos:
            print(self.dataframe_generator(cells_info))

import os
import datetime
import cv2
import numpy
import pandas

from .image_untils import ImageUntils

class SturcturAnalysis:
    def __init__(self, cells_infos):
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
        dataframes = []
        for cells_info in self.cells_infos:
            dataframe = self.dataframe_generator(cells_info)
            dataframe = PostProcessing(dataframe).run()
            dataframes.append(dataframe)

        return dataframes


class PostProcessing:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.df_dict = self.df_to_dict(dataframe)

    def df_to_dict(self, df):
        df_json = df.to_json(orient='index')
        return eval(df_json)
    
    def type_check(self, df_dict):
        header_rN = len(df_dict)//2
        predict_header = [list(dict(pp).values())
                        for pp in list(df_dict.values())[0:header_rN]]
        if predict_header[0][0] == '(empty_cell)':
            predict_header[0][0] = '#col0row0#'
        judge_list = []
        for i, row in enumerate(predict_header):
            if '(empty_cell)' in row:
                judge_list.append(i)

        return 'simple' if len(judge_list) == 0 else 'complex'
    
    def simple_post_processing(self, df_dict):
        newDictKeys = list(dict(list(df_dict.values())[0]).values())
        newDictKeys = [pp.replace(' ', '_') for pp in newDictKeys]
        if newDictKeys[0][0] == '(empty_cell)':
            newDictKeys[0][0] = '#col0row0#'

        newDictValues = [None]*len(newDictKeys)
        for n, v in enumerate(newDictValues):
            newDictValues[n] = [None]*(len(df_dict)-1)
        for i, (key, value) in enumerate(list(df_dict.items())[1:]):
            for ii, (col, info) in enumerate(list(dict(value).items())):
                newDictValues[int(col[3:])][int(key)-1] = info
        newDict = dict(zip(newDictKeys, newDictValues))

        return newDict
    
    def transform(self, df_dict):
        df_list = [None]*len(list(dict(list(df_dict.values())[0]).values()))
        for n, v in enumerate(df_list):
            df_list[n] = [None]*(len(df_dict))
        for i, (key, value) in enumerate(list(df_dict.items())):
            for ii, (col, info) in enumerate(list(dict(value).items())):
                df_list[int(col[3:])][int(key)] = info
        if df_list[0][0] == '(empty_cell)':
            df_list[0][0] = '#col0row0#'

        return df_list
    
    def vertical_melting(self, df_list):
        row_list = []
        number = len(df_list[0])
        for i in range(number):
            row_i = []
            for row in df_list:
                row_i.append(row[i])
            row_list.append(row_i)
        sch_row = []
        for i in range(len(row_list)-1):
            zwei_rows = [row_list[i][m]+row_list[i+1][m] for m in range(len(row_list[i]))]
            result = all(['(empty_cell)' in cell for cell in zwei_rows[2:]])
            if result == True:
                if i-1 not in sch_row:
                    sch_row.append(i)
        if len(sch_row) != 0:
            for n in sch_row:
                for spalt in df_list:
                    spalt[n] = (spalt[n] + '_' + spalt[n+1]
                                ).replace('_(empty_cell)', '').replace('(empty_cell)_', '').replace('-_', '-')
            df_list_new = [None]*len(df_list)
            for i in range(len(df_list_new)):
                df_list_new[i] = [cell for ii, cell in enumerate(df_list[i]) if ii-1 not in sch_row]
    
            return df_list_new
        else:
            return df_list
        
    def line_title_melting(self, df_list):
        halb_rN = len(df_list[0])//3 + 1
        if '(empty_cell)' in df_list[0][halb_rN:] + df_list[1][halb_rN:]:
            for i, ind in enumerate(df_list[0]):
                df_list[0][i] = (df_list[0][i] + '_' + df_list[1][i]
                                ).replace('_(empty_cell)', '').replace('(empty_cell)_', '')
            del df_list[1]

        return df_list
    
    def header_analysis(self, df_list):
        row_list = []
        number = len(df_list[0])
        for i in range(number):
            row_i = []
            for row in df_list:
                row_i.append(row[i])
            row_list.append(row_i)
        str_set = ['(empty_cell)']
        voll_row_pre = []
        empty_row = []
        voll_row = []
        for i, row in enumerate(row_list):
            result = any([cell in str_set for cell in row])
            if result == True:
                empty_row.append(i)
            else:
                if i != 0 and i != 1: 
                    voll_row_pre.append(i)
        if len(voll_row_pre) != 0:
            for i,n in enumerate(voll_row_pre):
                if i!=len(voll_row_pre)-1:
                    if voll_row_pre[i] == voll_row_pre[i+1]-1:
                        voll_row.append(n)
            if len(voll_row) != 0:
                zeile_nummer = voll_row[0]
            else:
                zeile_nummer = voll_row_pre[0]
        else:
            zeile_nummer = number//3 + 1

        return zeile_nummer, empty_row, row_list
    
    def header_melting(self, df_list, zeile_nummer, empty_row, row_list):
        if 0 in empty_row:
            empty_cell = []
            # hier sind die indexs von der Zellen, die aufgefüllt werden muss.
            first_header = []
            # hier sind die indexs von primär header, die coppiert werden muss.

            for col, value_row0 in enumerate(row_list[0][1:]):
                if value_row0 == '(empty_cell)':
                    empty_cell.append(col+1)
                else:
                    first_header.append(col+1)

            header_zeile = row_list[:zeile_nummer]
            headercol_list = []
            for i in range(len(row_list[0])):
                col_i = []
                for row in header_zeile:
                    col_i.append(row[i])
                headercol_list.append(col_i)
            str_set = ['(empty_cell)']
            no_quali = []
            for i, col in enumerate(headercol_list):
                # if all cell are empty, result = True
                result = all([cell in str_set for cell in col[1:]])
                if result == True:
                    no_quali.append(i)
            for col in first_header:
                if col+1 not in empty_cell:
                    no_quali.append(col)

            first_header = [pri for pri in first_header if pri not in no_quali]
            empty_cell = [emp for emp in empty_cell if emp not in no_quali]

            b = 1
            while len(first_header) != 0:
                delete = []
                for col in first_header:
                    # expand to the right
                    if col+b in empty_cell:
                        headercol_list[col+b][0] = headercol_list[col][0]
                        del empty_cell[empty_cell.index(col+b)]
                    else:
                        # basierend auf PositionCorrection ist die rechte Seite unbedingt leer, falls nicht leer ist es ein Fehler.
                        delete.append(col)
                for col in first_header:    
                    if col-b in empty_cell:
                        headercol_list[col-b][0] = headercol_list[col][0]
                        del empty_cell[empty_cell.index(col-b)]
                    else:
                        # In der vorherigen for-Schleife wurde die leere Zelle auf der rechten Seite der Titelzelle gefüllt.
                        # Wenn die linke Seite zu diesem Zeitpunkt nicht gefüllt werden kann, sollte die Titelzelle deaktiviert werden.
                        delete.append(col)
                delete = list(set(delete))
                for col in delete:
                    del first_header[first_header.index(col)]
                b += 1
                if b >= 20:
                    break

        else:
            header_zeile = row_list[:zeile_nummer]
            headercol_list = []
            for i in range(len(row_list[0])):
                col_i = []
                for row in header_zeile:
                    col_i.append(row[i])
                headercol_list.append(col_i)

        newDictKeys = ['_'.join(item) for item in headercol_list]
        newDictKeys = [pp.replace('(empty_cell)_', '') for pp in newDictKeys]
        newDictKeys = [pp.replace('_(empty_cell)', '') for pp in newDictKeys]
        newDictKeys = [pp.replace('#col0row0#_', '') for pp in newDictKeys]
        newDictKeys = [pp.replace(' ', '_') for pp in newDictKeys]
        newDictKeys = [pp.replace('-', '_') for pp in newDictKeys]
        # in keys should not have  ' ', '-' etc.
        newDictKeys = [pp.replace('.', '_') for pp in newDictKeys]
        newDictKeys = [pp.replace('__', '_') for pp in newDictKeys]
        
        newDictValues = [col[zeile_nummer:] for col in df_list]
        newDict = dict(zip(newDictKeys, newDictValues))

        return newDict

    
    def run(self):
        if self.type_check(self.df_dict) == 'simple':
            return self.simple_post_processing(self.df_dict)
        
        else:
            df_list = self.transform(self.df_dict)
            i = 0
            while i < 3:
                df_list = self.vertical_melting(df_list)
                i += 1
            m = 0
            while m < 2:
                df_list = self.line_title_melting(df_list)
                m += 1
            
            df_dict = self.header_melting(df_list, *self.header_analysis(df_list))
            return pandas.DataFrame(df_dict)
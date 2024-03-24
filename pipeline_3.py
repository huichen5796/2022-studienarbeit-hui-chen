import cv2
import os

from lib import *

def pipeline_3(file_path):
    result_save_path = file_path.replace('\\','/').replace('store_image_finder', 'store_done_history').rsplit('.', 1)[0]
    image = cv2.imread(file_path, 0)
    image_rotate = TiltCorrector(image).rotate()
    square_image = SquareNormaler(image_rotate).size_normaler()
    cv2.imwrite(f'{result_save_path}_norm.png', square_image)
    table_boundRects = TableFinder(square_image, 'densenet','cpu').run()
    column_boundRects_relativ = ColumnFinder(square_image, 'densenet', 'cpu', table_boundRects).run()
    cell_boundRects_relativ = CellFinder(square_image, table_boundRects).run()
    cell_boundRects_relativ_withLabel = CellLabelGenerator(column_boundRects_relativ, cell_boundRects_relativ).run()
    cells_infos = TableOcr(square_image, table_boundRects, cell_boundRects_relativ_withLabel).run()
    dataframes = SturcturAnalysis(cells_infos).run()
    # for i, dataframe in enumerate(dataframes):
    #     ElasticUntils('table', image_name=file_path, table_index=i).save(dataframe)

    return {
        "image_norm_path": f'{result_save_path}_norm.png',
        "table_zones": table_boundRects,
        "column_boundRects_relativ": column_boundRects_relativ,
        "cells_infos": cells_infos
    }
    


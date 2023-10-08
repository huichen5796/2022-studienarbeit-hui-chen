import cv2
import os

from lib import *
from conf import *

def pipeline_3(file_path):
    image = cv2.imread(file_path, 0)
    image_rotate = TiltCorrector(image).rotate()
    square_image = SquareNormaler(image_rotate).size_normaler()
    table_boundRects = TableFinder(square_image, 'densenet','cpu').run()
    column_boundRects_relativ = ColumnFinder(square_image, 'densenet', 'cpu', table_boundRects).run()
    cell_boundRects_relativ = CellFinder(square_image, table_boundRects).run()
    cell_boundRects_relativ_withLabel = CellLabelGenerator(column_boundRects_relativ, cell_boundRects_relativ).run()
    cells_infos = TableOcr(square_image, table_boundRects, cell_boundRects_relativ_withLabel).run()
    dataframes = SturcturAnalysis(cells_infos).run()
    for i, dataframe in enumerate(dataframes):
        ElasticUntils('table', image_name=os.path.basename(file_path), table_index=i).save(dataframe)

if __name__ == '__main__':
    FILE_PATH = 'Abbildungen/test2.PNG'
    pipeline_3(FILE_PATH)

### for test:
    # print(ElasticUntils('table').save_excel(saveRoot='', tableId='all'))
    # print(ElasticUntils('table').save_excel(saveRoot='', imageId="test2.PNG"))
    # print(ElasticUntils('table', os.path.basename(FILE_PATH), 0).search())
    # print(ElasticUntils('table').search(search_all=True))
    # ElasticUntils('table', os.path.basename(FILE_PATH), 0).detele()

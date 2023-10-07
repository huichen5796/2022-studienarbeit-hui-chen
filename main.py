from lib import CellFinder, CellLabelGenerator, TableOcr, ColumnFinder, ElasticUntils, ImageUntils, LineDetector, LogWriter, SquareNormaler, SturcturAnalysis, TableFinder, TiltCorrector
from conf import *
import cv2

def main():
    image = cv2.imread('Development/imageTest/test6.png', 0)
    image_rotate = TiltCorrector(image).rotate()
    square_image = SquareNormaler(image_rotate).size_normaler()
    table_boundRects = TableFinder(square_image, 'densenet','cpu').run()
    column_boundRects_relativ = ColumnFinder(square_image, 'densenet', 'cpu', table_boundRects).run()
    cell_boundRects_relativ = CellFinder(square_image, table_boundRects).run()
    cell_boundRects_relativ_withLabel = CellLabelGenerator(column_boundRects_relativ, cell_boundRects_relativ).run()
    cells_infos = TableOcr(square_image, table_boundRects, cell_boundRects_relativ_withLabel).run()
    SturcturAnalysis(cells_infos).run()

main()
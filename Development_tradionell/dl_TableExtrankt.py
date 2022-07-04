import cv2
import numpy as np


from Development_DL.Arbeitbreich_DL.PositioningTable import PositionTable
from Development_tradionell.Arbeitbreich.TableExtract import TiltCorrection, DeletLines, GetCell, HorizonalAlignment, ReadCell, GetInfoDict, WriteData

def Main(path):
    image = cv2.imread(path, 0)
    image_rotate = TiltCorrection(image) # gray
    image = cv2.cvtColor(image_rotate, cv2.COLOR_GRAY2BGR) # 3 channel
    table_boundRect = PositionTable(image) 

    table_zone = [None]*len(table_boundRect)
    for i, (x,y,w,h) in enumerate(table_boundRect):
    
        table_zone[i] = np.ones((h, w, 1))

        table_zone[i] = image_rotate[x:(x+h),(y):(y+w)]
    
    print('image '+ path + 'has ' + len(table_zone) +'table(s)')

    for table in table_zone:
        table_ol = DeletLines(table)
        location = GetCell(image_rotate)
        location = HorizonalAlignment(location)

        list_info = ReadCell(location, image_rotate)
        dict_info = GetInfoDict(list_info)
        WriteData(dict_info)

# Hier ist noch eine einfache Gliederung, 
# die Tabelle im Bild wird gelesen und in die Datenbank geschrieben, 
# aber sie kann noch nicht die Verwandtschaft zweier Tabellen aus demselben Bild widerspiegeln.




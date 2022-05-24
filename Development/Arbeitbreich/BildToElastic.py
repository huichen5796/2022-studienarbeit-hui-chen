'''

- write Words in Elasticsearch
- Elasticsearch in Excal


'''

from table_extract import GetInfoList
import time
print('------------start--------------')
print(time.ctime(time.time()))
GetInfoList(r'Development\imageTest\textandtablewinkel.png', 5, 3, 1)
print(time.ctime(time.time()))
GetInfoList(r'Development\imageTest\textandtablewinkel.png', 5, 3, 2)
print(time.ctime(time.time()))
print('------------finish-------------')
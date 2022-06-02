'''

- write Words in Elasticsearch


'''

from table_extract import GetInfoDict
import time
import json
from elasticsearch import Elasticsearch

es = Elasticsearch()

def WriteData(dict_info):
    i = 1
    es.index(index='table%s' %i, doc_type = '_doc', body = dict_info)


def Search(index):
    """ 
    Searches for data in ES-index
    op: operator can be "and" (e.g. must match "Husten AND Fieber") or "or" (e.g. "Husten OR Fieber")
    """

    reqBody = {
        "size": 1000, # no. of hits that will be sent
        "query": {
            "match_all": {} # gives back all entries in ES-index
        }
    }
    
    res = es.search(index=index, body=reqBody)

    data_print = json.dumps(res, indent=4, ensure_ascii=False).encode('utf8') # preperation for pretty-print: encoding with utf-8 for "ä, ö, etc."
    # print(data_print.decode()) # pretty-print with indent level
    return data_print.decode()


#dict_info = GetInfoDict(r'Development\imageTest\textandtablewinkel.png', 5, 3, 1)
#WriteData(dict_info)
#data_print = Search('table1')
#print(data_print)

es.indices.delete(index='table1', ignore=[400, 404]) # deletes whole index 




######## für Test ################
def ZeitVergleichung():

    print('------------start--------------')
    print(time.ctime(time.time()))
    dict_info = GetInfoDict(r'Development\imageTest\textandtablewinkel.png', 5, 3, 1)
    print(dict_info)
    print(time.ctime(time.time()))
    dict_info = GetInfoDict(r'Development\imageTest\textandtablewinkel.png', 5, 3, 2)
    print(dict_info)
    print(time.ctime(time.time()))
    print('------------finish-------------')
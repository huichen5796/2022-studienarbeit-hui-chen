from elasticsearch import Elasticsearch
es = Elasticsearch()
import pandas as pd
import os
import json
from functions import Search

def SaveExcel():
    writer = pd.ExcelWriter('Development\\saveTable.xlsx', engine='xlsxwriter')

    uniqueId_list = []
    res = Search('table', 'all')
    res = json.loads(res)['hits']['hits']
    for re in res:
        uniqueId_list.append(re['_source']['uniqueId'])
    uniqueId_list = list(set(uniqueId_list))
    
    for i, label in enumerate(uniqueId_list):

        table = []
        results = Search('table', str(label))
        results = json.loads(results)['hits']['hits']
        for result in results:
            del result['_source']['uniqueId']
            del result['_source']['fileName']
            table.append(result['_source'])
        df = pd.DataFrame(table)
        df.index.name = label
        df.to_excel(writer, sheet_name='Sheet%s'%i)

    writer.save()

if __name__ == '__main__':
    SaveExcel()
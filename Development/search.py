
from elasticsearch import Elasticsearch
es = Elasticsearch()
import pandas as pd
import json
from functions import Search
results = Search('table', 'all')
#print(results)

# show in dataframe
results = json.loads(results)
writer = pd.ExcelWriter('Development\\saveTable.xlsx', engine='xlsxwriter')
for i,result in enumerate(results['hits']['hits']):

    df = pd.DataFrame(result['_source']['content'])

    df.to_excel(writer, sheet_name='Sheet%s'%i)
    #print('--------------------')
    table_label = result['_source']['uniqueId']
    #print(table_label)
    #print(df)

writer.save()
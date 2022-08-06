'''
from elasticsearch import Elasticsearch
es = Elasticsearch()
import pandas as pd
import json
from functions import Search
results = Search('table', 'all')

# show in dataframe
results = json.loads(results)
for result in results['hits']['hits']:
    df = pd.DataFrame(result['_source']['content'])
    print('--------------------')
    table_label = result['_source']['uniqueId']
    print(table_label)
    print(df)

'''

empty_cell = [2, 3, 5, 6]
first_header = [4]
header_rows = [['KW 2022', 'Delta', '(empty_cell)', '(empty_cell)', 'Omikron', '(empty_cell)', '(empty_cell)'],
               ['(empty_cell)', '(empty_cell)', 'BA.1', 'BA.2', 'BA.3', 'BA.4', 'BA.5']]


x = []
while '(empty_cell)' in header_rows[0][1:]:
    for col in empty_cell:

        if col-1 in first_header:
            header_rows[0][col] = header_rows[0][col-1]
            first_header.append(col)
            print(header_rows)

        else:
            if col+1 in first_header:
                header_rows[0][col] = header_rows[0][col+1]
                first_header.append(col)
                print(header_rows)

print(header_rows)


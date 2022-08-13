
from elasticsearch import Elasticsearch
es = Elasticsearch()
import pandas as pd
import json
from functions import Search
results = Search('table', 'all')
# print(results)

# show in dataframe
results = json.loads(results)
for result in results['hits']['hits']:
    df = pd.DataFrame(result['_source']['content'])
    print('--------------------')
    table_label = result['_source']['uniqueId']
    print(table_label)
    print(df)


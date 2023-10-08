from elasticsearch import Elasticsearch, helpers
import os
import datetime
import json
import pandas

class ElasticUntils:
    def __init__(self, elastic_index, image_name=None, table_index=None):
        self.es = Elasticsearch()
        self.elastic_index = elastic_index
        self.label = f"table_{str(table_index)}_of_{image_name}"
        self.image_name = image_name

    def save(self, df):
        df_dict = eval(df.to_json(orient='index'))
        values = []
        for key, value in list(df_dict.items()):
            values.append(dict(value))
        for bulk in values:
            actions = []
            bulk["uniqueId"] = self.label.lower()
            bulk["fileName"] = self.image_name

            actions.append(bulk)
            helpers.bulk(self.es, actions, index='table')
    
    def save_excel(self, saveRoot, tableId=None, imageId=None):
        uniqueId_list = []
        if imageId:
            savePath = saveRoot + os.path.splitext(imageId)[0] + '.xlsx'
            writer = pandas.ExcelWriter(savePath, engine='xlsxwriter')
            self.image_name = imageId
            res = self.search(search_image=True)
            res = json.loads(res)['hits']['hits']
            for re in res:
                uniqueId_list.append(re['_source']['uniqueId'])
            uniqueId_list = list(set(uniqueId_list))
        else:
            savePath = saveRoot + os.path.splitext(tableId)[0] + '.xlsx'
            writer = pandas.ExcelWriter(savePath, engine='xlsxwriter')
            if tableId == 'all':
                res = self.search(search_all=True)
                res = json.loads(res)['hits']['hits']
                for re in res:
                    uniqueId_list.append(re['_source']['uniqueId'])
                uniqueId_list = list(set(uniqueId_list))
            else:
                uniqueId_list.append(tableId)
        
        for i, label in enumerate(uniqueId_list):
            table = []
            self.label = label
            results = self.search(search_all=False)
            results = json.loads(results)['hits']['hits']
            for result in results:
                del result['_source']['uniqueId']
                del result['_source']['fileName']
                table.append(result['_source'])
            df = pandas.DataFrame(table)
            # df.index.name = label
            df.to_excel(writer, sheet_name=label[0:30])
        writer.save()
        
        return savePath
        

    def search(self, search_all = None, search_image = None):
        if search_all:
            reqBody = {
                "size": 1000,  # no. of hits that will be sent
                "query": {
                    "match_all": {}  # gives back all entries in ES-index
                }
            }
        else:
            if search_image:
                reqBody = {
                    "size": 1000,  # No. of hits that will be sent
                    "query": {
                        "match": {
                            'fileName': {
                                'query': self.image_name,
                                'operator': 'and'
                            }
                        }
                    }
                }
            else:
                reqBody = {
                    "size": 1000,  # No. of hits that will be sent
                    "query": {
                        "match": {
                            'uniqueId': {
                                'query': self.label,
                                'operator': 'and'
                            }
                        }
                    }
                }

        res = self.es.search(index=self.elastic_index, body=reqBody)
        data_print = json.dumps(res, indent=4, ensure_ascii=False).encode('utf8')
        
        return data_print.decode()
    
    def detele(self, delete_all = None, delete_image = None):
        if delete_all:
            self.es.indices.delete(index=self.elastic_index, ignore=[400,404])
        else:
            if delete_image:
                reqBody = {
                    "size": 1000,  # No. of hits that will be sent
                    "query": {
                        "match": {
                            'fileName': {
                                'query': self.image_name,
                                'operator': 'and'
                            }
                        }
                    }
                }
            else:
                reqBody = {
                    "size": 1000,  # No. of hits that will be sent
                    "query": {
                        "match": {
                            'uniqueId': {
                                'query': self.label,
                                'operator': 'and'
                            }
                        }
                    }
                }
            res = self.es.search(index=self.elastic_index, body=reqBody)
            for hit in res['hits']['hits']:
                document_id = hit['_id']
                self.es.delete(index=self.elastic_index, doc_type="_doc", id=document_id)

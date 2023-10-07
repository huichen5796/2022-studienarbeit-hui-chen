from elasticsearch import Elasticsearch, helpers
import pandas

class ElasticUntils:
    def __init__(self, elastic_index, file_name, table_index):
        self.es = Elasticsearch()
        self.elastic_index = elastic_index
        self.label = f"table_{str(table_index+1)}_of_{file_name}"
        self.file_name = file_name

    def save(self, df):
        df_dict = eval(df.to_json(orient='index'))
        values = []
        for key, value in list(df_dict.items()):
            values.append(dict(value))
        for bulk in values:
            actions = []
            bulk["uniqueId"] = self.label.lower()
            bulk["fileName"] = self.file_name

            actions.append(bulk)
            helpers.bulk(self.es, actions, index='table')


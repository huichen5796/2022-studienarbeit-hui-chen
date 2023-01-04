import pandas as pd

def GetDataframe(list_info, label_list, tablesize):

    keys = ['col%s' % (s+1) for s in range(tablesize[1])]

    values = [None]*len(keys)
    for i, key in enumerate(keys):
        col_info = []
        index = []
        for m in range(len(label_list)):
            if key in label_list[m]:
                col_info.append(list_info[m])
                index.append(label_list[m][0])

        values[i] = pd.Series(col_info, index=index)
        values[i] = values[i].to_dict()  # Deduplizierung
        values[i] = pd.Series(values[i])

    dict_info = dict(zip(keys, values))
    # print(dict_info)
    df = pd.DataFrame(dict_info)
    #df = df.fillna('(empty_cell)')
    print('\n')
    print(df)
    print('\n')
    return df

GetDataframe(['column1', 'row1', 'value1'], [["row1", "col2"], ["row2", "col1"], ["row2", "col2"]], [2,2])
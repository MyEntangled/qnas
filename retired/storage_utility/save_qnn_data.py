import pandas as pd
import numpy as np
from itertools import compress

def create_df(columns,indices,data=None):
    df = pd.DataFrame(data=data, index=indices, columns=columns)

    # alert duplicated indices
    df_index = df.index
    if not df_index.is_unique:
        overlap = df_index[df_index.duplicated()].unique()
        raise ValueError(f"Indexes have overlapping values: {overlap}")

    return df

def insert_to_df(df1, df2):
    df = df1.append(df2, ignore_index=False, verify_integrity=True)
    return df

def concat_df(*many_df):
    df = pd.concat(many_df, ignore_index=False, verify_integrity=True)
    return df

def insert_columns(df, columns_to_add):
    for column in columns_to_add:
        df[column] = np.nan
    return df

def remove_columns(df, columns_to_del, remove_not_nan_columns=False):
    d = df[columns_to_del].isna() # if each entry is nan (entry-wise)
    if d.all(axis=None):
        df = df.drop(columns_to_del, axis=1)
    else:
        if remove_not_nan_columns:
            df = df.drop(columns_to_del, axis=1)
        else:
            not_nan_idx = ~d.all(axis=0)
            not_nan_cols = list(compress(columns_to_del, not_nan_idx))
            raise PermissionError(f"Columns with some value-assigned entries {not_nan_cols}")

    return df


if __name__ == '__main__':
    columns = ['filename','type','expr']
    indices_1 = ['qnn_1','qnn_2','qnn_3']
    indices_2 = ['qnn_4','qnn_5','qnn_6']
    indices_3 = ['qnn_7','qnn_8','qnn_9']
    zeros = np.zeros((3,3))
    ones = np.ones((3,3))

    df1 = create_df(columns,indices_1,zeros)
    df2 = create_df(columns,indices_2)
    df3 = create_df(columns,indices_3,ones)

    df_a = insert_to_df(df1, df2)
    #print(df_a)

    df_b = concat_df(df1,df2,df3)
    print(df_b)

    df_c = insert_columns(df_b,['extracol_1', 'extracol_2', 'extracol_3'])
    print(df_c)

    df_d = remove_columns(df_c, ['extracol_1', 'extracol_2', 'expr'], remove_not_nan_columns=True)
    #df_d = remove_columns(df_c, df_c.columns[:-1], remove_not_nan_columns=False)
    
    print(df_d)


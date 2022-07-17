from multiprocessing import Pool
import numpy as np
import pandas as pd
from helper_functions import read_data


pd.options.display.width = None
pd.options.display.max_columns = None

data_raw = read_data('data/')
cat_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']


def prepare_chunks_cust(df, columns):
    """
    Prepares chunks by customers
    :param df: pandas dataframe
    :param columns: columns to be used
    :return: list of pandas dataframes
    """
    cust_unique_ids = df['customer_ID'].unique()
    cust_ids_split = np.array_split(cust_unique_ids, 12)
    ready_chunks = []

    for c_ids in cust_ids_split:
        sub = df[df['customer_ID'].isin(c_ids)][['customer_ID'] + columns]
        ready_chunks.append(sub)
    return ready_chunks


chunks_to_process = prepare_chunks_cust(data_raw, cat_features)
print(chunks_to_process[0].head(20))

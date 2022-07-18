import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import repeat


def read_data(directory='', train=True, sample=False, cust_ratio=0.1):
    """
    Args:
    train - bool - True if to read a train file
    sample - bool - True if to draw a sample
    cust_ratio - float - a ratio of customers to be sampled
    """

    if train:
        file = 'train.parquet'
    else:
        file = 'test.parquet'

    df = pd.read_parquet(f'{directory}/{file}')

    print(f"Database shape: {df.shape}")
    if sample:
        n_customers = df['customer_ID'].nunique()
        no_of_cust = int(n_customers * cust_ratio)
        cust_ids = np.random.choice(df['customer_ID'].unique(), no_of_cust)
        df = df[df['customer_ID'].isin(cust_ids)]
        print(f'Rows in sampled database: {df.shape[0]}')
    return df


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


def _ewmt(df, hl):
    """
    Calculates EWM for a chunk
    Args:
        df: pandas database
        hl: integer, halflife value

    Returns: pandas dataframe

    """
    df_new = df.ewm(halflife=hl).mean()
    #df_new.columns = [f'{df_new.columns[0]}_ewm{hl}']
    df_new = df_new.add_suffix(f'ewm{hl}')
    return df_new


def calc_ewm(chunks, periods=(2, 4)):
    """
    Calculates EWM
    Args:
        chunks: list, contains pandas dataframes
        periods: list, contains periods for EWM

    Returns: pandas dataframe
    """
    ewm_results = []
    for t in periods:
        p1 = Pool(cpu_count())
        ewm_results.append(p.starmap(_ewmt, zip(chunks, repeat(t))))
        p1.close()
        p1.join()

    rows_joined = []
    for c in ewm_results:
        ewm_results = pd.concat(c)
        rows_joined.append(ewm_results)
    final = pd.concat(rows_joined, axis=1)
    return final


def _cat_stat(df):
    """
    Calculates categorical statistics for a chunk
    Args:
        df: pandas dataframe

    Returns: pandas dataframe with statistics

    """
    cat_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']
    data_cat_agg = df.groupby("customer_ID")[cat_features].agg(['count','first', 'last', 'nunique'])
    data_cat_agg.columns = ['_'.join(x) for x in data_cat_agg.columns]
    return data_cat_agg


def calc_categorical_stats(chunks):
    """
    Calculates categorical statistics for all chunks
    Args:
        chunks: list of pandas dataframe

    Returns: pandas dataframe with calculated statistics

    """
    p2 = Pool(cpu_count())
    results = p.map(_cat_stat, chunks)
    p2.close()
    p2.join()

    results = pd.concat(results)
    return results

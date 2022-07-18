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
    df_new.columns = [f'{df_new.columns[0]}_ewm{hl}']
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
        p = Pool(cpu_count())
        ewm_results.append(p.starmap(_ewmt, zip(chunks, repeat(t))))
        p.close()
        p.join()

    rows_joined = []
    for c in ewm_results:
        ewm_results = pd.concat(c)
        rows_joined.append(ewm_results)
    final = pd.concat(rows_joined, axis=1)
    return final

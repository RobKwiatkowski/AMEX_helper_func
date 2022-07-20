import pandas as pd
import numpy as np
import gc
from multiprocessing import Pool, cpu_count
from itertools import repeat


def read_data(directory='', train=True, sample=False, cust_ratio=0.2):
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


def prepare_chunks_cust(df, columns, n_chunks=12):
    """
    Prepares chunks by customers
    :param df: pandas dataframe
    :param columns: columns to be used
    :param n_chunks: number of chunks to be generated

    :return: list of pandas dataframes
    """
    cust_unique_ids = df['customer_ID'].unique()
    cust_ids_split = np.array_split(cust_unique_ids, n_chunks)
    ready_chunks = []

    for c_ids in cust_ids_split:
        sub = df[df['customer_ID'].isin(c_ids)][columns]
        ready_chunks.append(sub)
    return ready_chunks


def _ewmt(chunk, periods):
    """
    Calculates EWM for a chunk
    Args:
        df: pandas database
        periods: list, periods of halflife value

    Returns: pandas dataframe

    """
    results = []
    cust_ids = chunk['customer_ID']
    for t in periods:
        chunk = chunk.ewm(halflife=t).mean()
        chunk = chunk.add_suffix(f'ewm{t}')
        results.append(pd.concat([cust_ids, chunk], axis=1))
    df = pd.concat(results)
    return df


def calc_ewm(chunks, periods=(2, 4)):
    """
    Calculates EWM
    Args:
        chunks: list, contains pandas dataframes
        periods: list, contains periods for EWM

    Returns: pandas dataframe
    """
    ewm_results = []
    p1 = Pool(cpu_count())
    ewm_results.append(p1.starmap(_ewmt, zip(chunks, repeat(periods))))
    p1.close()
    p1.join()

    gc.collect()
    final = pd.concat(ewm_results[0])
    del ewm_results
    return final


def _cat_stat(df):
    """
    Calculates categorical statistics for a chunk
    Args:
        df: pandas dataframe

    Returns: pandas dataframe with statistics

    """
    cat_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']
    data_cat_agg = df.groupby("customer_ID")[cat_features].agg(['count', 'first', 'last', 'nunique'])
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
    results = p2.map(_cat_stat, chunks)
    p2.close()
    p2.join()

    results = pd.concat(results)
    return results


def prepare_date_features(df):

    def _take_first_col(series): return series.values[0]
    def _last_2(series): return series.values[-2] if len(series.values) >= 2 else -127
    def _last_3(series): return series.values[-3] if len(series.values) >= 3 else -127

    # Converting S_2 column to datetime column
    df['S_2'] = pd.to_datetime(df['S_2'])

    # How many rows of records does each customer has?
    df['rec_len_date'] = df.loc[:, ['customer_ID', 'S_2']].groupby(by=['customer_ID'])['S_2'].transform('count')

    # Encode the 1st statement and the last statement time
    df['S_2_first'] = df.loc[:, ['customer_ID', 'S_2']].groupby(by=['customer_ID'])['S_2'].transform('min')
    df['S_2_last'] = df.loc[:, ['customer_ID', 'S_2']].groupby(by=['customer_ID'])['S_2'].transform('max')

    # For how long(days) the customer is receiving the statements
    df['S_2_period'] = (df[['customer_ID', 'S_2']].groupby(by=['customer_ID'])['S_2'].transform('max') -
                        df[['customer_ID', 'S_2']].groupby(by=['customer_ID'])['S_2'].transform('min')).dt.days

    # Days Between 2 statements
    df['days_between_statements'] = \
        df[['customer_ID', 'S_2']].sort_values(by=['customer_ID', 'S_2']).groupby(by=['customer_ID'])['S_2'].transform(
        'diff').dt.days
    df['days_between_statements'] = df['days_between_statements'].fillna(0)
    df['days_between_statements_mean'] = df[['customer_ID', 'days_between_statements']].sort_values(
        by=['customer_ID', 'days_between_statements']).groupby(by=['customer_ID']).transform('mean')
    df['days_between_statements_std'] = df[['customer_ID', 'days_between_statements']].sort_values(
        by=['customer_ID', 'days_between_statements']).groupby(by=['customer_ID']).transform('std')
    df['days_between_statements_max'] = df[['customer_ID', 'days_between_statements']].sort_values(
        by=['customer_ID', 'days_between_statements']).groupby(by=['customer_ID']).transform('max')
    df['days_between_statements_min'] = df[['customer_ID', 'days_between_statements']].sort_values(
        by=['customer_ID', 'days_between_statements']).groupby(by=['customer_ID']).transform('min')
    df['S_2'] = (df['S_2_last'] - df['S_2']).dt.days

    # Difference between S_2_last(max) and S_2_last
    df['S_2_last_diff_date'] = (df['S_2_last'].max() - df['S_2_last']).dt.days

    # Difference between S_2_first(min) and S_2_first
    df['S_2_first_diff_date'] = (df['S_2_first'].min() - df['S_2_first']).dt.days

    # Get the (day,month,year) and drop the S_2_first because we can't directly use them
    df['S_2_first_dd_date'] = df['S_2_first'].dt.day
    df['S_2_first_mm_date'] = df['S_2_first'].dt.month
    df['S_2_first_yy_date'] = df['S_2_first'].dt.year

    df['S_2_last_dd_date'] = df['S_2_last'].dt.day
    df['S_2_last_mm_date'] = df['S_2_last'].dt.month
    df['S_2_last_yy_date'] = df['S_2_last'].dt.year

    agg_df = df.groupby(by=['customer_ID']).agg({'S_2': ['last', _last_2, _last_3],
                                                 'days_between_statements': ['last', _last_2, _last_3]})

    agg_df.columns = [i + '_' + j for i in ['S_2', 'days_between_statements'] for j in ['last', 'last_2', 'last_3']]
    df = df.groupby(by=['customer_ID']).agg(_take_first_col)
    df = df.merge(agg_df, how='inner', left_index=True, right_index=True)
    df = df.drop(['S_2', 'days_between_statements', 'S_2_first', 'S_2_last_x'], axis=1)

    return df

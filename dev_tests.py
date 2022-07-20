if __name__ == '__main__':
    import pandas as pd
    import helper_functions as hf

    pd.options.display.width = None
    pd.options.display.max_columns = 15

    data_raw = hf.read_data('data', sample=0.2)

    cat_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']
    num_features = [c for c in data_raw.columns if c not in cat_features]

    # numerical stats - EWM
    chunks_to_process = hf.prepare_chunks_cust(data_raw, num_features)
    df_ewms = hf.calc_ewm(chunks_to_process, periods=[2, 4, 6])
    print('writing EWMs data...')
    df_ewms.to_parquet('df_ewm.gzip')
    del chunks_to_process

    # datetime stats
    df = data_raw.loc[:, ['customer_ID', 'S_2']]
    df_date = hf.prepare_date_features(df)
    print('writing datetime stats...')
    df_date.to_csv('df_date.csv')

    # categorical stats
    chunks_to_process = hf.prepare_chunks_cust(data_raw, ['customer_ID']+cat_features)
    df_cats = hf.calc_categorical_stats(chunks_to_process)
    print('writing categorical stats...')
    df_cats.to_csv('df_cats.csv')

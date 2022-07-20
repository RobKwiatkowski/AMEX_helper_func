if __name__ == '__main__':
    import gc
    import numpy as np
    import pandas as pd
    import helper_functions as hf

    pd.options.display.width = None
    pd.options.display.max_columns = 15

    data_raw = hf.read_data('data', sample=True, cust_ratio=0.2)

    c_ids = data_raw['customer_ID'].unique()
    pd.DataFrame(c_ids).to_csv('outputs/cust_ids.csv', index=False)

    cat_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']
    num_features = [c for c in data_raw.columns if c not in cat_features+['customer_ID']]

    payment_cols = [col for col in list(data_raw.columns) if 'P_' in col]
    delinquency_cols = [col for col in list(data_raw.columns) if 'D_' in col]
    spend_cols = [col for col in list(data_raw.columns) if 'S_' in col]
    balance_cols = [col for col in list(data_raw.columns) if 'B_' in col]
    risk_cols = [col for col in list(data_raw.columns) if 'R_' in col]
    print(payment_cols, '\n', spend_cols)

    # numerical stats - EWM
    ewm_cols = [c for c in spend_cols if c in num_features]
    print(f'number of columns for EWMs: {len(ewm_cols)}')

    for i, split_cols in enumerate(np.array_split(ewm_cols, 4)):
        split_cols = list(split_cols)
        split_cols.insert(0, 'customer_ID')

        chunks_to_process = hf.prepare_chunks_cust(data_raw, split_cols, n_chunks=6)
        df_ewms = hf.calc_ewm(chunks_to_process, periods=[2, 4])
        print(f'writing EWMs{i} data...')
        df_ewms.to_parquet(f'outputs/df_ewm{i}.gzip')

        del chunks_to_process, df_ewms
        gc.collect()

    # datetime stats
    df = data_raw.loc[:, ['customer_ID', 'S_2']]
    df_date = hf.prepare_date_features(df)
    print('writing datetime stats...')
    df_date.to_csv('outputs/df_date.csv')
    del df_date
    gc.collect()

    # categorical stats
    chunks_to_process = hf.prepare_chunks_cust(data_raw, ['customer_ID']+cat_features)
    df_cats = hf.calc_categorical_stats(chunks_to_process)
    print('writing categorical stats...')
    df_cats.to_csv('outputs/df_cats.csv')
    del df_cats
    gc.collect()

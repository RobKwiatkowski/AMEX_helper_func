if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from helper_functions import read_data, prepare_chunks_cust, calc_ewm, calc_categorical_stats, prepare_date_features

    pd.options.display.width = None
    pd.options.display.max_columns = None

    data_raw = read_data('data', sample=0.2)
    print(data_raw.shape)
    cat_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']
    num_features = [c for c in data_raw.columns if c not in cat_features]

    # # numerical stats - EWM
    # chunks_to_process = prepare_chunks_cust(data_raw, num_features)
    # df_ewm = calc_ewm(chunks_to_process, periods=[2, 4, 6])
    # print(df_ewm.head())
    # del chunks_to_process
    df = data_raw.loc[:, ['customer_ID', 'S_2']]
    df_date = prepare_date_features(df, cat_features, num_features)
    print(df_date.head())


    # # categorical stats
    # chunks_to_process = prepare_chunks_cust(data_raw, cat_features)
    # df_ewm = calc_categorical_stats(chunks_to_process)
    # print(df_ewm.head())

if __name__ == '__main__':
    import pandas as pd
    import glob

    pd.options.display.width = None
    pd.options.display.max_columns = 25

    data = pd.read_csv('data/train_labels.csv').set_index('customer_ID')
    print(data.shape)

    files = glob.glob('outputs/*.parquet')
    ewms = pd.read_parquet(files[0])
    ewms.set_index('customer_ID', inplace=True)
    for f in files[1:]:
        new_def = pd.read_parquet(f)
        ewms = pd.concat([ewms, new_def.set_index('customer_ID')], axis=1, join='inner')
        print(ewms.shape)

    data = pd.concat([data, ewms], axis=1, join='outer')
    print(data.shape)

    df_date = pd.read_csv('outputs/df_date.csv')
    data = pd.concat([data, df_date.set_index('customer_ID')], axis=1, join='outer')
    print(data.shape)

    df_cats = pd.read_csv('outputs/df_cats.csv')
    data = pd.concat([data, df_cats.set_index('customer_ID')], axis=1, join='outer')

    print(data.shape)
    print(data.head())

    data.to_parquet('outputs/train_df.parquet')



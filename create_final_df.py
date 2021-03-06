if __name__ == '__main__':
    import pandas as pd
    import glob
    import os

    def create_final(train=False, sample=False):
        pd.options.display.width = None
        pd.options.display.max_columns = 25

        if train:
            data = pd.read_csv('data/train_labels.csv').set_index('customer_ID')
        else:
            if sample:
                data = pd.read_csv('outputs/cust_ids.csv').set_index('customer_ID')
            else:
                data = pd.read_csv('data/sample_submission.csv').set_index('customer_ID').drop('prediction', axis=1)

        files = glob.glob('outputs/*.parquet')
        ewms = pd.read_parquet(files[0])
        ewms.set_index('customer_ID', inplace=True)
        print(data.shape)
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

        print(data.describe())
        if train:
            data.to_parquet('outputs/final_dfs/train_df.parquet')
        else:
            data.to_parquet('outputs/final_dfs/test_df.parquet')

    def check_readiness():
        names = [os.path.basename(x) for x in glob.glob('outputs/final_dfs/*')]
        print(f'Files found: {names}')
        tmp_df = pd.read_parquet('outputs/final_dfs/train_df.parquet')
        train_cols = tmp_df.columns
        print(f'Columns in a training set: {len(train_cols)}')

        tmp_df = pd.read_parquet('outputs/final_dfs/test_df.parquet')
        test_cols = tmp_df.columns
        print(f'Columns in a testing set: {len(test_cols)}')

        del tmp_df


    create_final(train=True)
    check_readiness()

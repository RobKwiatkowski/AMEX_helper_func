import pandas as pd
import numpy as np
import glob

pd.options.display.width = None
pd.options.display.max_columns = 15

files = glob.glob('outputs/*.gzip')
data = pd.read_parquet(files[0])
data.set_index('customer_ID', inplace=True)
for f in files[1:]:
    new_def = pd.read_parquet(f, engine='fastparquet')
    new_def.set_index('customer_ID', inplace=True)
    data = pd.concat([data, new_def], axis=1)
data.reset_index(inplace=True)

print(data.head())

cust_ids = pd.read_csv('outputs/cust_ids.csv')
cust_ids = cust_ids.values.flat
data_ids = data['customer_ID'].unique().flat
print(np.shape(cust_ids), type(cust_ids))
print(np.shape(data_ids), type(data_ids))

print(np.array_equal(data['customer_ID'].unique(), cust_ids))

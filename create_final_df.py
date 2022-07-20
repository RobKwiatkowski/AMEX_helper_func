import pandas as pd
import glob

pd.options.display.width = None
pd.options.display.max_columns = 15

files = glob.glob('outputs/*.gzip')
data = pd.read_parquet(files[0])
for f in files[1:]:
    new_def = pd.read_parquet(f, engine='fastparquet')
    data = pd.concat([data, new_def], axis=1)

print(data.head())


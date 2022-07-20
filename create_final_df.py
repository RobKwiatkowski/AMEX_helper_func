import pandas as pd

df1 = pd.read_parquet("df_ewm.gzip", engine='fastparquet')
df2 = pd.read_csv("df_cats.csv")
df3 = pd.read_csv("df_date.csv")
print(df1.head())
print(df2.head())
print(df3.head())

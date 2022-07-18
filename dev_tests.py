from multiprocessing import Pool, cpu_count

import pandas as pd
from helper_functions import read_data, prepare_chunks_cust


pd.options.display.width = None
pd.options.display.max_columns = None

data_raw = read_data('data', sample=0.2)
cat_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']

chunks_to_process = prepare_chunks_cust(data_raw, cat_features)
print(chunks_to_process[0].head(5))
del data_raw

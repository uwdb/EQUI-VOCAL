import csv
from collections import defaultdict
import statistics
import pandas as pd

df = pd.read_table("data/products.tsv", dtype={"upc": int, "upc_descr": str, "product_group_descr": str})
df_select = df.iloc[:, [0, 2, 6]]

df_select['descr_cate'] = df_select[['upc_descr', 'product_group_descr']].apply(tuple, axis=1)
description_category_mode = df_select.groupby(['upc'])['descr_cate'].agg(lambda x: pd.Series.mode(x)[0]).reset_index()
description_category_mode['description'] = description_category_mode['descr_cate'].apply(lambda x: x[0])
description_category_mode['category'] = description_category_mode['descr_cate'].apply(lambda x: x[1])
output = description_category_mode.drop(['descr_cate'], axis=1)
output.to_csv("results/intermediate_results/updated_products.csv", index=False)
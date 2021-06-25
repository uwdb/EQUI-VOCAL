import os
import csv
import pandas as pd
import numpy as np

def descr_cate_price_log_scale():
    # Every UPC has unique (description, category); df1 contains the correct description
    df1 = pd.read_csv("results/intermediate_results/updated_products.csv", dtype={"upc": str, "description": str, "category": str})
    df_price = pd.read_csv("data/KiltsQuarterlyFinal.csv", dtype={"upc_actual_replace": str, "upc_descr": str, "product_group_descr": str})
    df2 = df_price.iloc[:, [0, 2, 8, 12, -4, -2]] # panel_year, quarter, upc_descr, product_group_descr, price, upc_actual_replace
    # Leave correct data only (upc with incorrect description is removed)
    merged = pd.merge(df1, df2, left_on=["upc", "description", "category"], right_on=["upc_actual_replace", "upc_descr", "product_group_descr"])
    # remove product with prices under 0.01
    quarter_data_price_filtered = merged[merged['price'] >= 0.01]
    
    merged_final = quarter_data_price_filtered[['upc', 'panel_year', 'quarter', 'price', 'category', 'description']]
    merged_final = merged_final.rename(columns={"panel_year": "year"})
    merged_final = merged_final.drop_duplicates()
    sort_csv = merged_final.sort_values(by=['price'])
    sort_csv = sort_csv.reset_index(drop=True)
    idx_list = range(len(sort_csv.index))
    sort_csv['index'] = pd.Series(idx_list)
    sort_csv.to_csv("results/intermediate_results/upc_year_quarter_price_cate_descr.csv", index=False)

if __name__ == '__main__':
    print("Step 2 begins...")
    descr_cate_price_log_scale()
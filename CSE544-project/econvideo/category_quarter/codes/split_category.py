'''
10 bins: LOG_BASE = 3
20 bins: LOG_BASE = 1.5
'''
import os
import csv
import pandas as pd
import numpy as np
from math import log

from scipy import stats

def remove_outlier(df, col):
  q1 = df[col].quantile(0.25)
  q3 = df[col].quantile(0.75)

  iqr = q3 - q1
  lower_bound  = q1 - (1.5  * iqr)
  upper_bound = q3 + (1.5 * iqr)

  out_df = df.loc[(df[col] > lower_bound) & (df[col] < upper_bound)]
  return out_df

def split_category_even():
    df = pd.read_csv("results/intermediate_results/upc_quarter_price_cate_descr.csv")
    category_list = df['category'].unique().tolist()
    if not os.path.exists("results/intermediate_results/data_per_category/"):
        os.makedirs("results/intermediate_results/data_per_category/")
    if not os.path.exists("results/intermediate_results/bin_per_category/"):
        os.makedirs("results/intermediate_results/bin_per_category/")
    predictor_idx = 0
    count_dict = {}
    predictor_idx_list = {}
    for category in category_list:
        if isinstance(category, str):
            for quarter in [1, 2, 3, 4]:
                filtered = df.loc[(df['category'] == category) & (df['quarter'] == quarter)].copy()
                filtered = remove_outlier(filtered, 'price')
                # filtered[(np.abs(stats.zscore(filtered['price'])) < 3)]
                filtered.reset_index(drop = True) 
                if filtered.shape[0] >= 10:
                    print(category, quarter, ' ==> ', predictor_idx)
                    predictor_idx_list[predictor_idx] = (category, quarter)
                    print(filtered.shape[0])
                    count_dict[predictor_idx] = filtered.shape[0]
                    with open("results/intermediate_results/bin_per_category/" + str(predictor_idx) + ".csv", 'w') as f:
                        field_names = ['bin', 'begin', 'end', 'count']
                        writer = csv.DictWriter(f, fieldnames = field_names)
                        writer.writeheader()

                        bin_list = []
                        num_per_bin = len(filtered.index) // 10
                        remainder = len(filtered.index) % 10
                        print(num_per_bin)
                        for i in range(remainder):
                            bin_list.extend([i] * (num_per_bin + 1))
                        for i in range(remainder, 10):
                            bin_list.extend([i] * num_per_bin)
                
                        # Compute the number of products in each bin
                        current_bin = 0 
                        count = 0
                        bin_count = [0] * 10
                        price_list = [filtered.iloc[0]['price']]
                        bin_list_idx = 0
                        for idx, row in filtered.iterrows():
                            filtered.loc[idx, 'bin'] = bin_list[bin_list_idx]
                            if bin_list[bin_list_idx] != current_bin:
                                print("bin:", current_bin, "count:", count)
                                price_list.append(row['price'])
                                bin_count[current_bin] = count
                                count = 0
                                current_bin += 1
                            count += 1
                            bin_list_idx += 1
                        print("bin:", current_bin, "count:", count)
                        bin_count[current_bin] = count
                        price_list.append(filtered.iloc[-1]['price'] + 1e-7)
                        filtered.to_csv("results/intermediate_results/data_per_category/" + str(predictor_idx) + ".csv", index=False)
                        for bin_idx in range(10):
                            writer.writerow({'bin': bin_idx, 'begin': price_list[bin_idx], 'end': price_list[bin_idx + 1], 'count': bin_count[bin_idx]})
                    predictor_idx += 1
    with open("results/intermediate_results/predictor_index.csv", 'w') as f:
    # with open("results/intermediate_results/category_index_remove_outliers.csv", 'w') as f:
        field_names = ['predictor_idx', 'category', 'quarter']
        writer = csv.DictWriter(f, fieldnames = field_names)
        writer.writeheader()
        for predictor_idx, category_quarter in predictor_idx_list.items():
            writer.writerow({'predictor_idx': predictor_idx, 'category': category_quarter[0], 'quarter': category_quarter[1]})

if __name__ == '__main__':
    # split_category_log_scale()
    print("Step 3 begins...")
    split_category_even()
import os
import csv
import pandas as pd
import numpy as np
import sys

def mjc_add_in_sample_value(predictor_idx):
    if not os.path.exists("results/intermediate_results/all_products_with_in_sample/"):
        os.makedirs("results/intermediate_results/all_products_with_in_sample/")
    # for i in range(20):
    # for i in [31]:
    all_prod_csv = f"results/intermediate_results/all_products/{predictor_idx}.csv"
    all_prod_train_csv = f"results/intermediate_results/all_products_train/{predictor_idx}.csv"
    output_csv = f"results/intermediate_results/all_products_with_in_sample/{predictor_idx}.csv"

    df_all = pd.read_csv(all_prod_csv)

    with open(all_prod_train_csv, mode='r') as train_csv_file:
        train_csv_reader = csv.reader(train_csv_file, delimiter=',')
        line_count = 0

        for row in train_csv_reader:
            if line_count != 0:
                idx = int(row[0])
                try:
                    df_all.loc[df_all['index'] == idx, 'In Sample'] = 1
                    # print("In Sample changed successfully!")
                except (KeyError, IndexError) as error:
                    continue
            else:
                line_count += 1
    
    df_all.to_csv(output_csv, index=False)

def mjc_all_products_only_per_category(predictor_idx):
    dir_name = "products_prediction/"
    if not os.path.exists("results/intermediate_results/" + dir_name):
        os.makedirs("results/intermediate_results/" + dir_name)
    # for i in range(20):
    # for i in [31]:
    df1 = pd.read_csv("results/intermediate_results/data_per_category/" + str(predictor_idx) + ".csv")
    df2 = pd.read_csv("results/intermediate_results/all_products_with_in_sample/" + str(predictor_idx) + ".csv")
    df2 = pd.merge(df1, df2, on="index")
    sort_csv = df2.sort_values(by=['index'])
    sort_csv = sort_csv.reset_index(drop=True)
    output_csv = sort_csv[['upc', 'description', 'category', 'True Price', 'True Bin', 'Predicted Price', 'Top1 Predicted Bin', 'Top2 Predicted Bin', 'In Sample']]
    output_csv.to_csv("results/intermediate_results/" + dir_name + str(predictor_idx) + ".csv", index=False)


if __name__ == '__main__':
    print("Step 5 begins...")
    predictor_idx = int(sys.argv[1])
    mjc_add_in_sample_value(predictor_idx)
    mjc_all_products_only_per_category(predictor_idx)
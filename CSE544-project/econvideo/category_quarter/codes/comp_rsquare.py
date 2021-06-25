'''
This script computes the R^2
'''

import csv
import sys
from scipy import stats

from sklearn.metrics import r2_score


if __name__ == '__main__':
    predictor_idx = int(sys.argv[1])
    with open("results/intermediate_results/rsquare.csv", mode='a') as csv_file:
        field_names = ['category #', 'in-sample', 'out-of-sample']
        csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
        # csv_writer.writeheader()

        # for i in range(20):
        # for i in [31]:
        print("category #" + str(predictor_idx))
        y_true_overall_in_sample = []
        y_pred_overall_in_sample = []

        y_true_overall_out_of_sample = []
        y_pred_overall_out_of_sample = []

        y_true_per_cate_in_sample = []
        y_pred_per_cate_in_sample = []

        y_true_per_cate_out_of_sample = []
        y_pred_per_cate_out_of_sample = []

        # csv_path = "data/mjc_all_products_1007.csv"
        csv_path = "results/intermediate_results/products_prediction/" + str(predictor_idx) + ".csv"

        with open(csv_path, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            for row in csv_reader:
                if line_count != 0:
                    true_price = float(row[3])

                    # # Whether overall predictor in sample
                    # if int(row[8]) == 1:
                    #     y_true_overall_in_sample.append(float(row[3]))
                    #     y_pred_overall_in_sample.append(float(row[4]))
                    # else:
                    #     y_true_overall_out_of_sample.append(float(row[3]))
                    #     y_pred_overall_out_of_sample.append(float(row[4]))
                    
                    # if int(row[13]) == 1:
                    #     y_true_per_cate_in_sample.append(float(row[3]))
                    #     y_pred_per_cate_in_sample.append(float(row[9]))
                    # else:
                    #     y_true_per_cate_out_of_sample.append(float(row[3]))
                    #     y_pred_per_cate_out_of_sample.append(float(row[9]))
                    
                    if int(row[8]) == 1:
                        y_true_per_cate_in_sample.append(float(row[3]))
                        y_pred_per_cate_in_sample.append(float(row[5]))
                    else:
                        y_true_per_cate_out_of_sample.append(float(row[3]))
                        y_pred_per_cate_out_of_sample.append(float(row[5]))

                else:
                    line_count += 1

        # _, _, r_value_1, _, _ = stats.linregress(y_true_overall_in_sample, y_pred_overall_in_sample)
        # print("Overall predictor, in-sample:", r_value_1 ** 2)

        # _, _, r_value_2, _, _ = stats.linregress(y_true_overall_out_of_sample, y_pred_overall_out_of_sample)
        # print("Overall predictor, out-of-sample:", r_value_2 ** 2)

        _, _, r_value_3, _, _ = stats.linregress(y_true_per_cate_in_sample, y_pred_per_cate_in_sample)
        print("Category predictor, in-sample:", r_value_3 ** 2)
        if y_true_per_cate_out_of_sample:
            _, _, r_value_4, _, _ = stats.linregress(y_true_per_cate_out_of_sample, y_pred_per_cate_out_of_sample)
        else:
            r_value_4 = 0
        print("Category predictor, out-of-sample:", r_value_4 ** 2)

        
        csv_writer.writerow({'category #': predictor_idx,
                                        'in-sample': r_value_3 ** 2, 
                                        'out-of-sample': r_value_4 ** 2
                                        })
'''
This script reads all data from KiltsQuarterlyFinal.csv, and writes (upc, price) pairs of all products to upc_price.csv.

Input file: 
    KiltsQuarterlyFinal.csv
    This gaint file contains abundant information of economics products. Each unique product can take up multiple rows because its prices are reported from different time (years, quarters, etc.).
    For every product, the price that is closest to 2nd quarter is chosen; if there is a tie, choose the price recorded in the most recent year.
Output file:
    upc_price.csv
    Each row represents a unique product. There are two columns in this csv: upc and price. 
    The data are sorted by price in ascending order.
'''

import csv
from collections import defaultdict
import statistics
import pandas as pd

print("Step 1 begins...")
upc_dict = {}
csvFile =  open('data/KiltsQuarterlyFinal.csv', 'r')
reader = csv.reader(csvFile)
count = 0
for item in reader:
    if count > 0: # Ignore the heading 
        price = item[-4]
        try:
            year = int(item[0])
            quarter = int(item[2])
            upc = int(item[-2])
            price = float(price)
            if price < 0.01:
                continue
            if not upc in upc_dict:
                upc_dict[upc] = []
            upc_dict[upc].append((year, abs(quarter - 2), price))
        except:
            continue
    count += 1

final_result = []
for key, value in upc_dict.items():
    # print('value', value)
    price = min(value, key = lambda t: (t[1], -t[0]))[2]
    # print('price', price)
    final_result.append({'upc': key, 'price': price})
df = pd.DataFrame(final_result)
sort_csv = df.sort_values(by=['price'])
sort_csv = sort_csv.reset_index(drop=True)
sort_csv.to_csv("results/intermediate_results/upc_price.csv", index=False)
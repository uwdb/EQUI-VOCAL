import pandas as pd
import numpy as np
import random
import csv
import torch
import os
from lstm import LSTM
from scipy.special import softmax
from dataset import get_train_val_loaders, prepare_sequence

EMBEDDING_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 128 # Not used

def get_emptiest_gpu():
	# Command line to find memory usage of GPUs. Return the one with most mem available.
	output = os.popen('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free').read().strip()
	mem_avail = [int(x.split()[2]) for x in output.split('\n')]
	return mem_avail.index(max(mem_avail))

torch.cuda.set_device(3)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# upc_info = "results/intermediate_results/upc_year_quarter_price_cate_descr.csv"
predictor_index = "results/intermediate_results/predictor_index.csv"
results = "results/intermediate_results/result_all.csv"

def pred(model, desc):
    model = str(model)
    _, _, word_len, word_dict= get_train_val_loaders(model)
    load_model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, word_len, BATCH_SIZE, 0.2, 0.8, 0.6)
    load_model.to(device)
    checkpoint = torch.load("results/checkpoints/" + model + "/epoch=25.checkpoint.pth.tar")
    # start_epoch = checkpoint['epoch']
    # stats = checkpoint['stats']
    load_model.load_state_dict(checkpoint['state_dict'])
    load_model.eval()
    #print(type(load_model))
    #desc = torch.tensor(desc, dtype=torch.int64)
    length = len(desc)
    desc_to_idx = torch.LongTensor(prepare_sequence([desc], word_dict))
    desc_to_idx = desc_to_idx.to(device)
    length = torch.LongTensor([length]).to(device)
    pred = load_model(desc_to_idx,length)
    pred = torch.nn.functional.softmax(pred, dim=1)
    top2 = torch.topk(pred, 2, dim=1)
    #print(top2)
    idx1 = top2[1][0][0].item()
    w1 = top2[0][0][0].item()
    idx2 = top2[1][0][1].item()
    w2 = top2[0][0][1].item()
    #print(idx1, w1, idx2, w2)
    price_range = pd.read_csv("results/intermediate_results/bin_per_category/" + model + ".csv")
    bin1 = price_range.loc[price_range['bin'] == idx1]
    price1 = (bin1['begin'] + bin1['end']) / 2
    price1 = price1.tolist()[0]
    bin2 = price_range.loc[price_range['bin'] == idx2]
    price2 = (bin2['begin'] + bin2['end']) / 2
    price2 = price2.tolist()[0]
    pred_price = (w1 * price1 + w2 * price2) / (w1 + w2)
    print("pred done 1.")
    return pred_price

with open(results, mode='w') as cvsfile:
    # Read all products 
    products = pd.read_csv(upc_info)
    upcs = products.drop_duplicates(subset=['upc'])

    # Read predictor info
    predictors = pd.read_csv(predictor_index)
    grouped_predictors = predictors.groupby(['category'])

    writer = csv.writer(cvsfile)
    writer.writerow(['upc', 'year', 'quarter', 'predict_price'])

    grouped_upcs = upcs.groupby(['category'])

    for group_name, df_group_upcs in grouped_upcs:
        print(group_name)
        df_group_predictors = grouped_predictors.get_group(group_name)
        for _, predictor_row in df_group_predictors.iterrows():
            model_id = predictor_row['predictor_idx']
            year = predictor_row['year']
            quarter = predictor_row['quarter']
            # Load model
            model_id = str(model_id)
            _, _, word_len, word_dict= get_train_val_loaders(model_id)
            load_model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, word_len, BATCH_SIZE, 0.2, 0.8, 0.6)
            load_model.to(device)
            checkpoint = torch.load("results/checkpoints/" + model_id + "/epoch=25.checkpoint.pth.tar")
            load_model.load_state_dict(checkpoint['state_dict'])
            load_model.eval()

            for _, upc_row in df_group_upcs.iterrows():
                upc = upc_row['upc']
                desc = upc_row['description']
                s1 = desc.split()
                s1.extend(group_name.split())

                length = len(s1)
                desc_to_idx = torch.LongTensor(prepare_sequence([s1], word_dict))
                desc_to_idx = desc_to_idx.to(device)
                length = torch.LongTensor([length]).to(device)
                pred = load_model(desc_to_idx,length)
                pred = torch.nn.functional.softmax(pred, dim=1)
                top2 = torch.topk(pred, 2, dim=1)
                #print(top2)
                idx1 = top2[1][0][0].item()
                w1 = top2[0][0][0].item()
                idx2 = top2[1][0][1].item()
                w2 = top2[0][0][1].item()
                #print(idx1, w1, idx2, w2)
                price_range = pd.read_csv("results/intermediate_results/bin_per_category/" + model_id + ".csv")
                bin1 = price_range.loc[price_range['bin'] == idx1]
                price1 = (bin1['begin'] + bin1['end']) / 2
                price1 = price1.tolist()[0]
                bin2 = price_range.loc[price_range['bin'] == idx2]
                price2 = (bin2['begin'] + bin2['end']) / 2
                price2 = price2.tolist()[0]
                pred_price = (w1 * price1 + w2 * price2) / (w1 + w2)

                writer.writerow([upc, year, quarter, pred_price])

        # for _, upc_row in df_group_upcs.iterrows():
        #     upc = upc_row['upc']
        #     desc = upc_row['description']
        #     s1 = desc.split()
        #     s1.extend(group_name.split())
        #     for _, predictor_row in df_group_predictors.iterrows():
        #         model_id = predictor_row['predictor_idx']
        #         year = predictor_row['year']
        #         quarter = predictor_row['quarter']
        #         pred_price = pred(model_id, s1)
        #         writer.writerow([upc, year, quarter, pred_price])


    # for _, row in upcs.iterrows():
    #     print(row)
    #     cat = row['category']
    #     upc = row['upc']
    #     desc = row['description']
    #     df_group = grouped_predictors.get_group(cat)
    #     s1 = desc.split()
    #     s1.extend(cat.split())
    #     for _, predictor_row in df_group.iterrows():
    #         model_id = predictor_row['predictor_idx']
    #         year = predictor_row['year']
    #         quarter = predictor_row['quarter']
    #         pred_price = pred(model_id, s1)
    #         writer.writerow([upc, year, quarter, pred_price])
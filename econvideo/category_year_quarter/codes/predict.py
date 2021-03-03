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

prod_info = f"results/intermediate_results/updated_products.csv"
output_csv = f"results/intermediate_results/sampled_products.csv"
results = f"results/intermediate_results/result.csv"
upc_info = f"results/intermediate_results/upc_year_quarter_price_cate_descr.csv"
predictor_index = "results/intermediate_results/predictor_index.csv"

def sample_product():
    products = pd.read_csv(upc_info)
    upcs = products['upc'].unique()
    print(len(upcs))

    idx = random.sample(list(upcs),k=1000)
    sampled_prod = products.loc[products.upc.isin(idx)]
    sampled = sampled_prod.sort_values(['upc','year','quarter'])
    sampled.to_csv(output_csv, index=False)

upc_year_quarter = pd.read_csv(upc_info)
#print(type(upc_year_quarter['upc'][0]))

def predecessor(new_row):
    cur_year = new_row[1]
    cur_quarter = new_row[2]
    if(cur_quarter!=str(1)):
        new_row.append(cur_year)
        new_row.append((int(cur_quarter)-1))
    else:
        new_row.append((int(cur_year)-1))
        new_row.append((4))

def following(new_row):
    cur_year = new_row[3]
    cur_quarter = new_row[4]
    if(cur_quarter!=str(4)):
        new_row.append(cur_year)
        new_row.append((int(cur_quarter)+1))
    else:
        new_row.append((int(cur_year)+1))
        new_row.append((1))

def check_available(new_row, cat):
    predictors = pd.read_csv(predictor_index)
    found = predictors.loc[(predictors['category']==cat)&(predictors['year'].astype(str)==str(new_row[5]))&(predictors['quarter'].astype(str)==str(new_row[6]))]
    idx = []
    if found.empty==False:
        new_row.append(1)
        idx.append(found['predictor_idx'])
    else:
        new_row.append(0)
    found = predictors.loc[(predictors['category']==cat)&(predictors['year'].astype(str)==str(new_row[7]))&(predictors['quarter'].astype(str)==str(new_row[8]))]
    if found.empty==False:
        new_row.append(1)
        idx.append(found['predictor_idx'])
    else:
        new_row.append(0)
    return idx

def predict(new_row, model_idx, desc):
    #print(new_row)
    #print(model_idx)
    #print(type(model_idx[0]))
    #print(type(desc))
    if int(new_row[9]) == 1:
        model = model_idx[0].to_string()
        model = model.split()[0]
        #print(model)
        #model_idx[0] = str(model_idx[0])
        #print(type(model))
        #print(a)
        if (os.path.exists("results/checkpoints/"+model+"/")):
            print("here1")
            _, _, word_len, word_dict= get_train_val_loaders(model)
            load_model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, word_len, BATCH_SIZE, 0.2, 0.8, 0.6)
            load_model.to(device)
            checkpoint = torch.load("results/checkpoints/"+model+"/epoch=25.checkpoint.pth.tar")
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
            price_range = pd.read_csv("results/intermediate_results/bin_per_category/"+model+".csv")
            bin1 = price_range.loc[price_range['bin']==idx1]
            price1 = (bin1['begin']+bin1['end'])/2
            price1 = price1.tolist()[0]
            bin2 = price_range.loc[price_range['bin']==idx2]
            price2 = (bin2['begin']+bin2['end'])/2
            price2 = price2.tolist()[0]
            #price1 = (price_range.loc[price_range['bin']==idx1]['begin']+price_range.loc[price_range['bin']==idx1]['end'])/2
            #price2 = (price_range.loc[price_range['bin']==idx2]['begin']+price_range.loc[price_range['bin']==idx2]['end'])/2
            pred_price = (w1*price1+w2*price2)/(w1+w2)
            new_row.append(pred_price)
            print("pred done 1.")
        else:
            new_row.append(-1)
    else:
        new_row.append(0)

    if int(new_row[10])==1:
        model = model_idx[-1].to_string()
        model = model.split()[0]
        #print(model)
        #model_idx[0] = str(model_idx[0])
        #print(type(model))
        #print(a)
        if (os.path.exists("results/checkpoints/"+model+"/")):
            print("here2")
            _, _, word_len, word_dict= get_train_val_loaders(model)
            load_model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, word_len, BATCH_SIZE, 0.2, 0.8, 0.6)
            load_model.to(device)
            checkpoint = torch.load("results/checkpoints/"+model+"/epoch=25.checkpoint.pth.tar")
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
            price_range = pd.read_csv("results/intermediate_results/bin_per_category/"+model+".csv")
            bin1 = price_range.loc[price_range['bin']==idx1]
            price1 = (bin1['begin']+bin1['end'])/2
            price1 = price1.tolist()[0]
            bin2 = price_range.loc[price_range['bin']==idx2]
            price2 = (bin2['begin']+bin2['end'])/2
            price2 = price2.tolist()[0]
            #price1 = (price_range.loc[price_range['bin']==idx1]['begin']+price_range.loc[price_range['bin']==idx1]['end'])/2
            #price2 = (price_range.loc[price_range['bin']==idx2]['begin']+price_range.loc[price_range['bin']==idx2]['end'])/2
            pred_price = (w1*price1+w2*price2)/(w1+w2)
            new_row.append(pred_price)
            print("pred done 2.")
        else:
            new_row.append(-1)
    else:
        new_row.append(0)
    


with open(results, mode='w') as cvsfile:
    writer = csv.writer(cvsfile)
    writer.writerow(['product_id','year_first_appear','quarter_first_appear','year_last_appear','quarter_last_appear','predecessor_year','predecessor_quarter','following_year','following_quarter','is_predecesor_model_available','is_following_model_available','predecessor_predicted_price','following_predicted_price'])
    with open(output_csv, mode='r') as sampled_prod:
        prod = csv.reader(sampled_prod, delimiter=',')
        #prod = products.sort_values(['upc','year','quarter'])
        line = 0
        cur_id = 0
        new_row = []
        year = 0
        quarter = 0
        s1 = []
        for row in prod:
            if line == 0:
                line = 1
                continue
            upc = row[0]
            cat = row[4]
            
            if upc != cur_id:
                if new_row:
                    new_row.append(year)
                    new_row.append(quarter)
                    predecessor(new_row)
                    following(new_row)
                    model_id = check_available(new_row, cat)
                    #print(model_id)
                    predict(new_row, model_id, s1)
                    writer.writerow(new_row)
                    new_row = []
                    desc = []
                cur_id = upc
                year = row[1]
                quarter = row[2]
                new_row.append(upc)
                new_row.append(row[1])
                new_row.append(row[2])
                #print(row[5])
                #print(row[4])
                s1 = row[5].split()
                s1.extend(row[4].split())
            else:
                year = row[1]
                quarter = row[2]
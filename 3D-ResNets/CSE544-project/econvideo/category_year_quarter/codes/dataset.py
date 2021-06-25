import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import config
from random import sample, choices
import random
import json

random.seed(42)
np.random.seed(42)
BATCH_SIZE = config("lstm.batch_size")

def get_train_val_loaders(category_idx):
    master_dataset = PriceDataset(category_idx)
    word_dict, word_len = master_dataset.get_dict()
    tr = MyDataset(master_dataset.X_train, master_dataset.y_train, master_dataset.true_train, master_dataset.upc_train)
    tr_loader = DataLoader(dataset=tr, batch_size=BATCH_SIZE, collate_fn=PadSequence(), shuffle=True, drop_last=False)
    va = MyDataset(master_dataset.X_val, master_dataset.y_val, master_dataset.true_val, master_dataset.upc_val)
    va_loader = DataLoader(dataset=va, batch_size=BATCH_SIZE, collate_fn=PadSequence(), shuffle=False, drop_last=False)
    
    return tr_loader, va_loader, word_len, word_dict

def get_train_val_loaders_no_sampling(category_idx):
    master_dataset = PriceDataset(category_idx)
    word_dict, word_len = master_dataset.get_dict()
    tr = MyDataset(master_dataset.X_train, master_dataset.y_train, master_dataset.true_train, master_dataset.upc_train)
    tr_loader = DataLoader(dataset=tr, batch_size=BATCH_SIZE, collate_fn=PadSequence(), shuffle=True, drop_last=False)
    all_dataset = PriceDataset_NoSampling(category_idx)
    all_dataset.X_all = prepare_sequence(all_dataset.X_all, word_dict)
    test = MyDataset(all_dataset.X_all, all_dataset.y_all, all_dataset.true_all, all_dataset.upc_all)
    test_loader = DataLoader(dataset=test, batch_size=BATCH_SIZE, collate_fn=PadSequence(), shuffle=False, drop_last=False)
    
    return test_loader, tr_loader, word_len, word_dict

def prepare_sequence(seq, to_ix):
    # -1 is used as padding token
    idxs = []
    for description in seq:
        idxs.append([to_ix[w] if w in to_ix.keys() else 1 for w in description])
    return idxs

class PadSequence:
    def __call__(self, batch):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        # Get each sequence and pad it
        sequences = [x[0] for x in sorted_batch]
        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])

        # create an empty matrix with padding tokens
        pad_token = 0
        longest_sent = lengths[0]
        batch_size = len(sequences)
        padded_X = np.ones((batch_size, longest_sent)) * pad_token

        # copy over the actual sequences
        for i, x_len in enumerate(lengths):
            sequence = sequences[i]
            padded_X[i, 0:x_len] = sequence[:x_len]
        sequences = torch.LongTensor(padded_X)
        # Don't forget to grab the labels of the *sorted* batch
        labels = torch.LongTensor([x[1] for x in sorted_batch])
        true_values = torch.FloatTensor([x[2] for x in sorted_batch])
        index_values = torch.LongTensor([x[3] for x in sorted_batch])
        return sequences, lengths, labels, true_values, index_values

class MyDataset(Dataset):
    def __init__(self, X, y, true_value, upc):
        """
        Reads in the necessary data from disk.
        """
        super().__init__()
        # Load in all the data we need from disk
        self.X = X
        self.y = y
        self.true_value = true_value
        self.upc = upc

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.int64), torch.tensor(self.y[idx], dtype=torch.int64), torch.tensor(self.true_value[idx], dtype=torch.float64), torch.tensor(self.upc[idx], dtype=torch.int64) 
    

class PriceDataset(Dataset):
    def __init__(self, category_idx):
        """
        Reads in the necessary data from disk.
        """
        super().__init__()

        np.random.seed(0)
    
        # Load in all the data we need from disk
        self.metadata = pd.read_csv(config("price_csv") + str(category_idx) + '.csv')
        self.X_train, self.X_val, self.y_train, self.y_val, self.true_train, self.true_val, self.upc_train, self.upc_val = self._load_data()
        self.word_to_ix = self._construct_dict()
        self._convert_data()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.true_value[idx]

    def _load_data(self):
        """
        Loads a single data partition from file. (either training or validating partition)
        """
        print("loading data...")
        
        X, y, true_value, temp_X_val, temp_y_val, temp_true_value = [], [], [], [], [], []
        upc_tr_list, upc_va_list = [], []
        val_X, val_y, val_true_value, val_upc = [], [], [], []

        bin_dict = {}
        for _, row in self.metadata.iterrows():
            if isinstance(row["category"], str) and isinstance(row["description"], str) and row["bin"] != config('lstm.num_classes'):
                if row["bin"] not in bin_dict:
                    bin_dict[row["bin"]] = []
                s1 = row["description"].split()
                s1.extend(row["category"].split())
                bin_dict[row["bin"]].append((s1, row["price"], row["index"])) # Word Embedding
                # bin_dict[row["bin"]].append((list(row["description"].upper()), row["price"])) # Character Embedding

        for price_bin, data_list in bin_dict.items():
            print("length:",len(data_list), "bin number:", price_bin)
            if len(data_list) < 4:
                data_list.append(data_list[-1])
            
            temp_X, temp_y, upc = zip(*data_list)

            X_train, X_val, true_train, true_val, upc_train, upc_val = train_test_split(temp_X ,temp_y, upc, test_size=0.25, random_state=42)
            
            NUM_SAMPLE = min(500, len(temp_X))

            for idx in np.random.choice(len(X_train), int(NUM_SAMPLE * 0.75), replace=(True if len(X_train) < int(NUM_SAMPLE * 0.75) else False)):
                X.append(X_train[idx])
                y.append(price_bin)
                true_value.append(true_train[idx])
                upc_tr_list.append(upc_train[idx])

            val_X.extend(X_val)
            val_y.extend([price_bin] * len(X_val))
            val_true_value.extend(true_val)
            val_upc.extend(upc_val)

        for idx in np.random.choice(len(val_X), int(NUM_SAMPLE * 0.25 * 10), replace=False):
            temp_X_val.append(val_X[idx])
            temp_y_val.append(val_y[idx])
            temp_true_value.append(val_true_value[idx])
            upc_va_list.append(val_upc[idx])

        return np.array(X), np.array(temp_X_val), np.array(y), np.array(temp_y_val), np.array(true_value), np.array(temp_true_value), np.array(upc_tr_list), np.array(upc_va_list) # X, y and true_value are train set; temp_X_val, temp_y_val and temp_true_value are validation set

    def _convert_data(self):
        self.X_train = prepare_sequence(self.X_train, self.word_to_ix)
        self.X_val = prepare_sequence(self.X_val, self.word_to_ix)

    def _construct_dict(self):
        word_to_ix = {}
        word_to_ix['<pad>'] = 0
        word_to_ix['<unk>'] = 1
        for description in self.X_train:
            for word in description:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        return word_to_ix

    def get_dict(self):
        return self.word_to_ix, len(self.word_to_ix)

class PriceDataset_NoSampling(Dataset):
    def __init__(self, category_idx):
        """
        Reads in the necessary data from disk.
        """
        super().__init__()

        np.random.seed(0)
    
        # Load in all the data we need from disk
        self.metadata = pd.read_csv(config("price_csv") + str(category_idx) + '.csv')
        self.X_all, self.y_all, self.true_all, self.upc_all = self._load_data() 

    def __len__(self):
        return len(self.X_all)

    def __getitem__(self, idx):
        return self.X_all[idx], self.y_all[idx], self.true_all[idx], self.upc_all[idx]

    def _load_data(self):
        """
        Loads a single data partition from file. (either training or validating partition)
        """
        print("loading data...")
        
        X_all, true_all, y_all, upc_all = [], [], [], []

        for _, row in self.metadata.iterrows():
            if isinstance(row["category"], str) and isinstance(row["description"], str) and row["bin"] != config('lstm.num_classes'):
                s1 = row["description"].split()
                s1.extend(row["category"].split())
                X_all.append(s1)
                y_all.append(row["bin"])
                true_all.append(row["price"])
                upc_all.append(row["index"])

        return np.array(X_all), np.array(y_all), np.array(true_all), np.array(upc_all)
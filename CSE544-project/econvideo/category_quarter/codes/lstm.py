import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from utils import config 
from torch.utils.data import Dataset, DataLoader, Subset, random_split, TensorDataset
from math import sqrt

NEG_INF = -10000
TINY_FLOAT = 1e-6
torch.manual_seed(1)

class LSTM(nn.Module):
    # Word embedding 
    def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size, dropout1, dropout2, dropout3):
        '''
        embedding_dim: dimension of each word vector (usually be more like 32 or 64 dimensional.)
        hidden_dim: dimension of hidden state
        vocab_size: 
        '''
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        # Input: (batch_size, variable_sequence_length)
        # Output: (batch_size, variable_sequence_length, embedding_dim)
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        print("vocab_size:",vocab_size,"embedding_dim:",embedding_dim)
        self.emb_drop = nn.Dropout(dropout1)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=2, dropout=dropout2, batch_first=True) # Input dimension is embedding_dim, output dim is hidden_dim
        # self.fc_att = nn.Linear(hidden_dim * 2, 1)
        # The linear layer that maps from hidden state space to category space
        self.fc1 = nn.Linear(hidden_dim * 3 , 256)
        self.fc_drop = nn.Dropout(dropout3)
        self.fc2 = nn.Linear(256 , config("lstm.num_classes"))
        self.init_weights()

    def init_weights(self):

        # initialize the parameters for [self.fc1, self.fc2, self.fc3]
        for fc in [self.fc1, self.fc2]:
            C_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 1/ sqrt(C_in))
            nn.init.constant_(fc.bias, 0.0)
        #
    def forward(self, X, length): # sentence has size (batch_size, max_sequence_length)
        X = self.word_embeddings(X) # Output has size (batch_size, max_sequence_length, embedding_dim)
        X = self.emb_drop(X)
        X = torch.nn.utils.rnn.pack_padded_sequence(X, length, batch_first=True)
        X, (hidden, _) = self.lstm(X) # X has size (batch_size, max_sequence_length, hidden_dim * 2); # hidden has size (1, batch_size, hidden_dim)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True, padding_value=0)

        avg_pool = F.adaptive_avg_pool1d(X.permute(0,2,1),1).view(X.size(0),-1)
        max_pool = F.adaptive_max_pool1d(X.permute(0,2,1),1).view(X.size(0),-1)
       
        z = torch.relu(self.fc1(torch.cat([hidden[-1],avg_pool,max_pool],dim=1)))   
        z = self.fc_drop(z)
        z = self.fc2(z)
        return z


def seq_mask(seq_len, max_len):
    """Create sequence mask.

    Parameters
    ----------
    seq_len: torch.long, shape [batch_size],
        Lengths of sequences in a batch.
    max_len: int
        The maximum sequence length in a batch.

    Returns
    -------
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.
    """

    idx = torch.arange(max_len).to(seq_len).repeat(seq_len.size(0), 1)
    mask = torch.gt(seq_len.unsqueeze(1), idx).to(seq_len)

    return mask

def mask_softmax(matrix, mask=None):
    """Perform softmax on length dimension with masking.

    Parameters
    ----------
    matrix: torch.float, shape [batch_size, .., max_len]
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.

    Returns
    -------
    output: torch.float, shape [batch_size, .., max_len]
        Normalized output in length dimension.
    """

    if mask is None:
        result = F.softmax(matrix, dim=-1)
    else:
        mask_norm = ((1 - mask) * NEG_INF).to(matrix)
        for i in range(matrix.dim() - mask_norm.dim()):
            mask_norm = mask_norm.unsqueeze(1)
        result = F.softmax(matrix + mask_norm, dim=-1)

    return result

def mask_mean(seq, mask=None):
    """Compute mask average on length dimension.

    Parameters
    ----------
    seq : torch.float, size [batch, max_seq_len, n_channels],
        Sequence vector.
    mask : torch.long, size [batch, max_seq_len],
        Mask vector, with 0 for mask.

    Returns
    -------
    mask_mean : torch.float, size [batch, n_channels]
        Mask mean of sequence.
    """

    if mask is None:
        return torch.mean(seq, dim=1)

    mask_sum = torch.sum(  # [b,msl,nc]->[b,nc]
        seq * mask.unsqueeze(-1).float(), dim=1)
    seq_len = torch.sum(mask, dim=-1)  # [b]
    mask_mean = mask_sum / (seq_len.unsqueeze(-1).float() + TINY_FLOAT)

    return mask_mean


def mask_max(seq, mask=None):
    """Compute mask max on length dimension.

    Parameters
    ----------
    seq : torch.float, size [batch, max_seq_len, n_channels],
        Sequence vector.
    mask : torch.long, size [batch, max_seq_len],
        Mask vector, with 0 for mask.

    Returns
    -------
    mask_max : torch.float, size [batch, n_channels]
        Mask max of sequence.
    """

    if mask is None:
        return torch.mean(seq, dim=1)

    mask_max, _ = torch.max(  # [b,msl,nc]->[b,nc]
        seq + (1 - mask.unsqueeze(-1).float()) * NEG_INF,
        dim=1)

    return mask_max
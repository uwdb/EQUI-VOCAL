'''
CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from cnn import CNN
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt, floor


class CNN(nn.Module):
    def __init__(self, input_shape, preprocessor, cnn_layer, fc_layer, dropout_rate):
        super(CNN, self).__init__()
        self.input_shape = input_shape
        self.color_mode = 3 if preprocessor == "ColorImage" else 1
        self.cnn_layer = cnn_layer
        self.fc_layer = fc_layer
        self.conv = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.conv.append(nn.Conv2d(self.color_mode, self.cnn_layer[0], 3, stride = 1, padding = 1))
        self.pool.append(nn.MaxPool2d((2, 2), stride=2))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        for i in range(1, len(self.cnn_layer)):
            self.conv.append(nn.Conv2d(self.cnn_layer[i-1], self.cnn_layer[i], 3, stride = 1, padding = 1))
            self.pool.append(nn.MaxPool2d((2, 2), stride=2))

        if self.input_shape[0] == self.input_shape[1]:
            final_size = floor(self.input_shape[0] / (2 ** len(self.pool))) ** 2 * self.cnn_layer[-1]
        else:
            final_size = floor(self.input_shape[0] / (2 ** len(self.pool))) * floor(self.input_shape[0] / (2 ** (len(self.pool) + 1))) * self.cnn_layer[-1]
            
        self.fc1 = nn.Linear(int(final_size), self.fc_layer[0])
        self.fc2 = nn.Linear(self.fc_layer[0], fc_layer[1])
        self.fc3 = nn.Linear(self.fc_layer[1], 2)
        # self.fc2 = nn.Linear(self.fc_layer, 1)
        self.init_weights()

    def init_weights(self):
        for conv in self.conv:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(3 * 3 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        # initialize the parameters for [self.fc1, self.fc2, self.fc3]
        for fc in [self.fc1, self.fc2, self.fc3]:
            C_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 1 / sqrt(2 * C_in))
            nn.init.constant_(fc.bias, 0.0)

    def forward(self, x):
        # forward pass
        z = None

        for conv, pool in zip(self.conv, self.pool):
            z = F.relu(conv(z if z is not None else x))
            z = pool(z)

        _, H, W, C = z.shape
        z = z.view(-1, C * H * W)
        z = F.relu(self.fc1(z))
        z = self.dropout(z)
        z = F.relu(self.fc2(z))
        #z = self.dropout(z)
        z = self.fc3(z)
        # z = torch.sigmoid(self.fc2(z))
        # z = z.squeeze()
        # print("z", z.shape)
        return z
import torch
import pandas as pd
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.loader import DataLoader
import random
from util import *

class BaselineModel(torch.nn.Module):
    def __init__(self, num_features=3, hidden_size=32, target_size=1,num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.num_layers = num_layers
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.conv = GATConv(self.num_features, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.target_size)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(x)
        x = self.conv(x, edge_index)
        # 假设x的维度是(batch_size, num_nodes, hidden_size)，将其调整为GRU输入需要的形状
        x = x.view(-1, 1, self.hidden_size)  # 调整形状为(batch_size, seq_len, input_size)
        # 通过GRU层
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 初始化隐藏状态
        x, _ = self.gru(x, h0)
        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = self.conv(x, edge_index) 
        x = self.linear(x)
        return (x) 

import os
import numpy as np
import scipy.sparse as sp 
import torch
from torch.utils.data import Dataset
import pandas as pd
import argparse
from tqdm import tqdm

path = "./data/all_data.csv"

class AirDataset(Dataset):
    def __init__(self, args):
        self.hist_len = args.hist_len
        self.pred_len = args.pred_len
        self.x, self.y = self._process_data()
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def _process_data(self):
        
        Airpollution = pd.read_csv(path)
        feature = ["2","3","4","5","6","7","8","9","10","11","12","13","14","15","16"] # 11 12 15 
        Airpollution = Airpollution.loc[:, feature]
        x_len = Airpollution.shape[0]
        node_num = 47
        all_timestamp = x_len // node_num
        
        sample_x = []
        sample_y = []
        # i is Dividing line
        for i in tqdm(range(self.hist_len, all_timestamp - self.pred_len)):
            x = np.float32(Airpollution.iloc[node_num * (i - self.hist_len) : node_num * i,:])
            y = np.float32(Airpollution.iloc[node_num * i:node_num*(i + self.pred_len), 1]) # 取label
            x = torch.from_numpy(x).view(x.shape[0]//node_num, -1, x.shape[1])
            y = torch.from_numpy(y).view(y.shape[0]//node_num, node_num)
            y = y[...,2]
            sample_x.append(x)
            sample_y.append(y)
            
        sample_x = torch.stack(sample_x)
        sample_y = torch.stack(sample_y)
        return sample_x, sample_y

# parser = argparse.ArgumentParser()

# parser.add_argument('--hist_len', default=24, type=int, help='a past period used for forecast')
# parser.add_argument('--pred_len', default=6, type=int, help='forecast how far in the futrue')
# args = parser.parse_args()
# Air = AirDataset(args)
# x, y = Air[0]
# print(x[1,...])
# print(y[0,...])
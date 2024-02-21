import sys
import os
import random

import time
from time import strftime, localtime
import argparse
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from data_utils import AirDataset
from trainer import Trainer
from model.GAT_ATCN import GAT_ATCN

t_start = time.time()

def get_logger(args):  
    logger = logging.getLogger()  
    logger.setLevel(logging.INFO)   
    logger.addHandler(logging.StreamHandler(sys.stdout))  
    if not os.path.exists(args.log_dir):  
        os.mkdir(args.log_dir, mode=0o777)   
    log_file = '{}-{}.log'.format(args.model_name, strftime("%Y-%m-%d_%H:%M:%S", localtime()))

    logger.addHandler(logging.FileHandler("%s/%s" % (args.log_dir, log_file)))  
    return logger  
   
def setup_seed(seed):    
    torch.manual_seed(seed)   
    torch.cuda.manual_seed_all(seed)   
    np.random.seed(seed)   
    random.seed(seed)  
    torch.backends.cudnn.deterministic = True  

model_classes = {
                 'GAT_ATCN': GAT_ATCN,
                 }

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data', type=str, help='The dictionary where the data is stored.')
parser.add_argument('--pollution', default='pollution', type=str, help='pollution, meteorology')
parser.add_argument('--log_dir', default='./log', type=str, help='The dictionary to store log files.')

parser.add_argument('--node_num', default=47, type=int, help='The number of cities')
parser.add_argument('--hist_len', default=24, type=int, help='a past period used for forecast')
parser.add_argument('--pred_len', default=6, type=int, help='forecast how far in the futrue')
parser.add_argument('--split_rate', default=0.8, type=float)

parser.add_argument('--model_name', default='GAT_ATCN', type=str, help=', '.join(model_classes))  # 添加选择哪个基线模型
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)

# GAT settings
parser.add_argument('--nfeat', default=15, type=int, help='input feature')
parser.add_argument('--nhid', default=8, type=int, help='GAT hidden unit nums')
parser.add_argument('--nout', default=6, type=int, help='GAT output nums')
parser.add_argument('--droupout', default=0.2, type=float)
parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--nheads', default=8, type=int)

# TCN settings
parser.add_argument('--tcn_hid_layer', default=[128, 64, 24])
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--shuffle', default=True, type=bool)
parser.add_argument('--cuda', default='0', type=str, help='gpu number')
parser.add_argument('--device', default=None, type=str, help='cpu, cuda')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device is None else torch.device(args.device)
print("choice cuda:{}".format(args.cuda))

logger = get_logger(args)

setup_seed(args.seed)

# data loader part
logger.info('\nLoading Dataset.')
AirpollutDataset = AirDataset(args)
train_size = int(0.8 * len(AirpollutDataset))
test_size = len(AirpollutDataset) - train_size
trainset, testset = random_split(AirpollutDataset, [train_size, test_size])  
train_dataloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=args.shuffle)
test_dataloader = DataLoader(dataset=testset, batch_size=args.batch_size)

# model train part
logger.info('\nTrain model')
args.model_class = model_classes[args.model_name]
model = args.model_class(args).to(args.device) 
optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
trainer = Trainer(args, logger, model, train_dataloader, test_dataloader, optimizer)
trainer.train()

t_end = time.time() 
# total time
logger.info('Training process took '+ str(round(t_end-t_start))+ ' secs.')
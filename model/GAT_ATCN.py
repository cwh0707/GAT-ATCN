import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.layer import GAT, TCN, AttentionWithFCN

class GAT_ATCN(nn.Module):
    def __init__(self, args):
        super(GAT_ATCN, self).__init__()
        self.args = args
        # GAT configure
        self.GAT = GAT(args.nfeat, args.nhid, args.nout, args.droupout, args.alpha, args.nheads)
        # TCN model
        self.TCN = TCN(args.node_num * args.nout, args.tcn_hid_layer)
        # Att-FCN model
        self.Att_FCN = AttentionWithFCN(args.tcn_hid_layer[-1], args.pred_len)

    
    def forward(self, input_x, adj):
        # # -----------------------------------
        """ GAT Part """
        x = self.GAT(input_x, adj)
        len = input_x.shape[0]
        x = x.view(len, -1).unsqueeze(0)
        x = x.permute(0, 2, 1)
        # # ------------------------------------

        # # ------------------------------------
        """ TCN Part """
        x = self.TCN(x)
        x = x.permute(0, 2, 1)
        # # ------------------------------------

        # # ------------------------------------
        """ ATT-FCN Part """
        x = self.Att_FCN(x)
        # # ------------------------------------
        return x
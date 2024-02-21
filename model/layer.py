import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W) # h.shape: (B, N, in_features), Wh.shape: (B, N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[1] # number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        return all_combinations_matrix.view(Wh.size()[0], N, N, 2 * self.out_features)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))  # 没有concat的地方进行elu激活
        return F.log_softmax(x, dim=2)


'''
    TCN part
'''

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding=1, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        # 确保out和res的长度相同
        if out.size(2) != res.size(2):
            res = F.pad(res, (0, out.size(2)-res.size(2)))  # 在最后一个维度的末尾填充1
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
'''
    注意力层
'''
    
class MultiplicativeAttention(nn.Module):
    def __init__(self, feature_dim, dropout=0.1):
        super(MultiplicativeAttention, self).__init__()
        self.scale = 1.0 / (feature_dim ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        # query, key, value: [batch_size, seq_len, feature_dim]
        # 计算查询和键的点积
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        # 应用softmax获取注意力权重
        attention_weights = self.softmax(attention_scores)
        attention_weights = self.dropout(attention_weights)
        # 计算加权的值
        weighted_output = torch.matmul(attention_weights, value)
        return weighted_output

class AttentionWithFCN(nn.Module):
    def __init__(self, tcn_output_dim, output_dim, dropout=0.1):
        super(AttentionWithFCN, self).__init__()
        self.attention = MultiplicativeAttention(tcn_output_dim, dropout=dropout)
        self.fc = nn.Linear(tcn_output_dim, output_dim)  # 全连接层定义
    
    def forward(self, x):
        tcn_out = x.squeeze()
        tcn_out = tcn_out.permute(1, 0)
        
        # 应用乘性注意力，这里我们使用tcn_out作为query, key, value
        attention_out = self.attention(tcn_out, tcn_out, tcn_out)
        
        # 假设我们对每个序列取平均作为最终特征表示
        # 也可以选择其他方法（如加权平均）来聚合序列
        final_feature = attention_out.mean(dim=1)
        
        # 通过全连接层获取最终的预测输出
        prediction = self.fc(final_feature)
        return prediction
    
# if __name__ == '__main__':
#     data_x = np.float32(np.random.randn(24, 47, 15))
#     data_y = np.float32(np.random.randn(6, 47))
#     gat = GAT(nfeat=15, nhid=64, nout=8, alpha=0.1, nheads=12, dropout=0.2)
#     tcn = TCN(47 * 8, [128, 64, 24])
#     adj = np.random.randint(2, size=(47, 47))
#     gat_out = gat(torch.from_numpy(data_x), torch.from_numpy(adj))
#     tcn_x= gat_out.view(24, -1).unsqueeze(0)
#     tcn_x = tcn_x.permute(0, 2, 1)
#     tcn_out = tcn(tcn_x)
#     tcn_out= tcn_out.permute(0, 2, 1)
#     att_fc = AttentionWithFCN(24, 6)
#     out = att_fc(tcn_out)
#     print(out)
#     print(out.shape)
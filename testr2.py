# -*- coding: utf-8 -*-

import numpy as np
import torch

# 生成两个随机的Tensor作为示例
tensor1 = torch.from_numpy(np.random.rand(5000, 6))
tensor2 = torch.from_numpy(np.random.rand(5000, 6))

# 计算均值
mean1 = torch.mean(tensor1,axis=0)
mean2 = torch.mean(tensor2,axis=0)

# 计算总平方和
TSS1 = torch.sum((tensor1 - mean1) ** 2, axis=0)
TSS2 = torch.sum((tensor2 - mean2) ** 2, axis=0)

# 计算残差平方和
RSS = torch.sum((tensor1 - tensor2) ** 2, axis=0)

# 计算R2相关系数
R2 = 1 - (RSS / (TSS1 + TSS2) / 2)

R2 = torch.mean(R2)
print("R2相关系数为：", R2)

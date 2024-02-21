import torch

def r2_score(target, outputs):
    # 计算均值
    mean1 = torch.mean(target, axis=0)
    mean2 = torch.mean(outputs, axis=0)

    # 计算总平方和
    TSS1 = torch.sum((target - mean1) ** 2, axis = 0)
    TSS2 = torch.sum((outputs - mean2) ** 2, axis = 0)

    # 计算残差平方和
    RSS = torch.sum((target - outputs) ** 2, axis = 0)

    # 计算R2相关系数
    R2 = 1 - (RSS / (TSS1 + TSS2) / 2)
    print(R2)
    R2 = torch.mean(R2)
    return R2
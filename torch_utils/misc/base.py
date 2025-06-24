import torch
import torch.nn as nn


def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')


# 初始化函数：对模型中的所有有参数的模块进行初始化
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He 初始化
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)  # 偏置初始化为0

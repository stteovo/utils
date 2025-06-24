import torch
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict

from models.backbone.ResNet import ResNet18


def load_checkpoint(checkpoint_path):
    """
    加载 .pth 文件并返回 checkpoint 字典。

    参数:
        checkpoint_path (str): .pth 文件的路径。

    返回:
        dict: 包含模型权重、优化器状态等信息的字典。
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    return checkpoint


def modify_model_weights(model, checkpoint):
    """
    修改模型中特定层的权重。

    参数:
        model (torch.nn.Module): 需要修改的 PyTorch 模型。
        checkpoint (dict): 包含模型权重的 checkpoint 字典。

    返回:
        torch.nn.Module: 修改后的模型。
    """
    # 将 checkpoint 中的权重加载到模型中
    model.load_state_dict(checkpoint['model_state_dict'])

    # 修改某个特定层的权重（例如，第一层卷积层）
    for name, param in model.named_parameters():
        if name == 'conv1.weight':
            print(f"Original {name} shape: {param.shape}")
            # 将所有权重设置为 0
            param.data.fill_(0)
            print(f"Modified {name} shape: {param.shape}")

    return model


def add_or_remove_layers(model, checkpoint):
    """
    添加或删除模型中的层，并迁移旧模型的权重到新模型中。

    参数:
        model (torch.nn.Module): 原始的 PyTorch 模型。
        checkpoint (dict): 包含模型权重的 checkpoint 字典。

    返回:
        torch.nn.Module: 修改后的模型。
    """

    class ModifiedResNet50(nn.Module):
        def __init__(self):
            super(ModifiedResNet50, self).__init__()
            self.resnet50 = models.resnet50(pretrained=False)
            # 添加一个新的全连接层，假设你想要将输出类别数改为 10
            self.fc = nn.Linear(1000, 10)

        def forward(self, x):
            x = self.resnet50(x)
            x = self.fc(x)
            return x

    # 创建新的模型实例
    new_model = ModifiedResNet50()

    # 将旧模型的权重加载到新模型中，忽略不匹配的层
    new_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    return new_model


def modify_optimizer(optimizer, checkpoint, new_lr=0.0001):
    """
    修改优化器的学习率和其他参数。

    参数:
        optimizer (torch.optim.Optimizer): 需要修改的优化器。
        checkpoint (dict): 包含优化器状态的 checkpoint 字典。
        new_lr (float): 新的学习率，默认为 0.0001。

    返回:
        torch.optim.Optimizer: 修改后的优化器。
    """
    # 加载优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 修改优化器的学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
        print(f"Updated learning rate to {new_lr}")

    return optimizer


def modify_epoch(checkpoint, new_epoch=10):
    """
    修改训练轮次。

    参数:
        checkpoint (dict): 包含训练轮次的 checkpoint 字典。
        new_epoch (int): 新的训练轮次，默认为 10。

    返回:
        dict: 修改后的 checkpoint 字典。
    """
    checkpoint['epoch'] = new_epoch
    print(f"Updated epoch to {new_epoch}")
    return checkpoint


def handle_multi_gpu_checkpoint(checkpoint):
    """
    处理多 GPU 训练的 .pth 文件，移除 `module.` 前缀。

    参数:
        checkpoint (dict): 包含模型权重的 checkpoint 字典。

    返回:
        dict: 修改后的 checkpoint 字典。
    """
    # 移除 `module.` 前缀
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k[7:] if k.startswith('module.') else k  # 移除 `module.`
        new_state_dict[name] = v

    # 更新 checkpoint 中的模型权重
    checkpoint['model_state_dict'] = new_state_dict
    return checkpoint


def change_key_names(checkpoint):
    # 移除 `module.` 前缀
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_k = 'backbone.' + k
        new_state_dict[new_k] = v

    # 更新 checkpoint 中的模型权重
    checkpoint = new_state_dict
    return checkpoint


def save_checkpoint(checkpoint, save_path):
    """
    保存修改后的 checkpoint 到指定路径。

    参数:
        checkpoint (dict): 包含模型权重、优化器状态等信息的字典。
        save_path (str): 保存 checkpoint 的路径。
    """
    print(f"Saving modified checkpoint to {save_path}...")
    torch.save(checkpoint, save_path)
    print("Checkpoint saved successfully.")


def main():
    # 定义文件路径
    checkpoint_path = '/data_ssd/ay/切脸数据/checkpoints/maskCompare_v3/latest_net_0.pth'
    # checkpoint_path = '/data_ssd/ay/切脸数据/checkpoints/MaskSmooth.pth'
    save_path = '/data_ssd/ay/切脸数据/checkpoints/MaskSmooth.pth'

    # 1. 加载 checkpoint
    checkpoint = load_checkpoint(checkpoint_path)

    # 2. 处理多 GPU 训练的 checkpoint（如果需要）
    # if any(k.startswith('module.') for k in checkpoint['model_state_dict']):
    #     checkpoint = handle_multi_gpu_checkpoint(checkpoint)

    # 3. 加载预训练的 ResNet-50 模型
    model = ResNet18(in_c=1)

    checkpoint = change_key_names(checkpoint)

    # 4. 修改模型权重
    # model = modify_model_weights(model, checkpoint)

    # 5. 添加或删除层（可选）
    # model = add_or_remove_layers(model, checkpoint)

    # 6. 创建优化器
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 7. 修改优化器状态
    # optimizer = modify_optimizer(optimizer, checkpoint, new_lr=0.0001)

    # 8. 修改训练轮次
    # checkpoint = modify_epoch(checkpoint, new_epoch=10)

    # 9. 更新 checkpoint 中的模型权重和优化器状态
    # checkpoint = model.state_dict()
    # checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    # 10. 保存修改后的 checkpoint
    save_checkpoint(checkpoint, save_path)


if __name__ == "__main__":
    main()
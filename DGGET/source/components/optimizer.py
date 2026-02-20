import logging
from collections import defaultdict
from typing import List
from omegaconf import DictConfig
import torch


def get_param_group_no_wd(model: torch.nn.Module, match_rule: str = None, except_rule: str = None):
    """
    根据模型的配置获取不同参数组，分别用于有权重衰减和无权重衰减的情况。

    输入：
    - model (torch.nn.Module): 要处理的模型。
    - match_rule (str, optional): 参数名中需要匹配的规则，默认为 None。
    - except_rule (str, optional): 参数名中需要排除的规则，默认为 None。

    输出：
    - List[DictConfig]: 包含两组参数的列表：无权重衰减和有权重衰减的参数组。
    - defaultdict: 参数类型计数器。
    """
    param_group_no_wd = []  # 无权重衰减的参数组
    names_no_wd = []  # 无权重衰减的参数名
    param_group_normal = []  # 有权重衰减的参数组

    type2num = defaultdict(lambda: 0)  # 参数类型计数器

    for name, m in model.named_modules():  # 遍历模型中的每个模块
        if match_rule is not None and match_rule not in name:
            continue
        if except_rule is not None and except_rule in name:
            continue
        if isinstance(m, torch.nn.Conv2d):
            if m.bias is not None:
                param_group_no_wd.append(m.bias)
                names_no_wd.append(name + '.bias')
                type2num[m.__class__.__name__ + '.bias'] += 1
        elif isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                param_group_no_wd.append(m.bias)
                names_no_wd.append(name + '.bias')
                type2num[m.__class__.__name__ + '.bias'] += 1
        elif isinstance(m, torch.nn.BatchNorm2d) \
                or isinstance(m, torch.nn.BatchNorm1d):
            if m.weight is not None:
                param_group_no_wd.append(m.weight)
                names_no_wd.append(name + '.weight')
                type2num[m.__class__.__name__ + '.weight'] += 1
            if m.bias is not None:
                param_group_no_wd.append(m.bias)
                names_no_wd.append(name + '.bias')
                type2num[m.__class__.__name__ + '.bias'] += 1

    for name, p in model.named_parameters():  # 遍历模型中的每个参数
        if match_rule is not None and match_rule not in name:
            continue
        if except_rule is not None and except_rule in name:
            continue
        if name not in names_no_wd:
            param_group_normal.append(p)

    params_length = len(param_group_normal) + len(param_group_no_wd)
    logging.info(f'Parameters [no weight decay] length [{params_length}]')
    return [{'params': param_group_normal}, {'params': param_group_no_wd, 'weight_decay': 0.0}], type2num


def optimizer_factory(model: torch.nn.Module, optimizer_config: DictConfig) -> torch.optim.Optimizer:
    """
    根据配置创建优化器。

    输入：
    - model (torch.nn.Module): 要优化的模型。
    - optimizer_config (DictConfig): 优化器的配置，包括学习率、权重衰减等参数。

    输出：
    - torch.optim.Optimizer: 创建的优化器实例。
    """
    parameters = {
        'lr': 0.0,  # 学习率初始值为 0.0，稍后会被更新
        'weight_decay': optimizer_config.weight_decay  # 权重衰减值
    }

    if optimizer_config.no_weight_decay:
        params, _ = get_param_group_no_wd(model,
                                          match_rule=optimizer_config.match_rule,
                                          except_rule=optimizer_config.except_rule)
    else:
        params = list(model.parameters())
        logging.info(f'Parameters [normal] length [{len(params)}]')

    parameters['params'] = params

    optimizer_type = optimizer_config.name
    if optimizer_type == 'SGD':
        parameters['momentum'] = optimizer_config.momentum
        parameters['nesterov'] = optimizer_config.nesterov
    return getattr(torch.optim, optimizer_type)(**parameters)


def optimizers_factory(model: torch.nn.Module, optimizer_configs: List[DictConfig]) -> List[torch.optim.Optimizer]:
    """
    根据多个配置创建优化器。

    输入：
    - model (torch.nn.Module): 要优化的模型。
    - optimizer_configs (List[DictConfig]): 优化器配置的列表。

    输出：
    - List[torch.optim.Optimizer]: 创建的优化器列表。
    """
    if model is None:
        return None
    return [optimizer_factory(model=model, optimizer_config=single_config) for single_config in optimizer_configs]

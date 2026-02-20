import bisect
import math
from typing import List
from omegaconf import DictConfig
import torch


class LRScheduler:
    """
    学习率调度器类，支持多种学习率调度策略。

    属性:
    - lr_config: DictConfig - 学习率调度的配置。
    - training_config: DictConfig - 训练的总体配置。
    - lr: float - 当前的学习率。
    """

    def __init__(self, cfg: DictConfig, optimizer_cfg: DictConfig):
        """
        初始化学习率调度器。

        输入：
        - cfg (DictConfig): 训练配置，包含总步数 (total_steps) 和每步学习率的配置。
        - optimizer_cfg (DictConfig): 优化器配置，包含学习率调度器的配置 (lr_scheduler) 和初始学习率 (lr)。

        输出：
        - 无
        """
        self.lr_config = optimizer_cfg.lr_scheduler  # 学习率调度配置
        self.training_config = cfg  # 训练配置
        self.lr = optimizer_cfg.lr  # 初始学习率

        assert self.lr_config.mode in [
            'step', 'poly', 'cos', 'linear', 'decay']  # 确保模式合法

    def update(self, optimizer: torch.optim.Optimizer, step: int):
        """
        更新学习率并应用到优化器的参数组。

        输入：
        - optimizer (torch.optim.Optimizer): 要更新的优化器。
        - step (int): 当前的训练步骤数。

        输出：
        - 无
        """
        lr_config = self.lr_config
        lr_mode = lr_config.mode  # 获取学习率调度模式
        base_lr = lr_config.base_lr  # 基础学习率
        target_lr = lr_config.target_lr  # 目标学习率

        warm_up_from = lr_config.warm_up_from  # 预热阶段的起始学习率
        warm_up_steps = lr_config.warm_up_steps  # 预热步骤数
        total_steps = self.training_config.total_steps  # 总训练步骤数

        assert 0 <= step <= total_steps  # 确保步骤数合法
        if step < warm_up_steps:
            current_ratio = step / warm_up_steps  # 计算当前预热比例
            self.lr = warm_up_from + (base_lr - warm_up_from) * current_ratio  # 线性插值计算学习率
        else:
            current_ratio = (step - warm_up_steps) / \
                            (total_steps - warm_up_steps)  # 计算当前比例
            if lr_mode == 'step':
                count = bisect.bisect_left(lr_config.milestones, current_ratio)  # 计算当前的里程碑数量
                self.lr = base_lr * pow(lr_config.decay_factor, count)  # 计算学习率
            elif lr_mode == 'poly':
                poly = pow(1 - current_ratio, lr_config.poly_power)  # 多项式衰减
                self.lr = target_lr + (base_lr - target_lr) * poly  # 计算学习率
            elif lr_mode == 'cos':
                cosine = math.cos(math.pi * current_ratio)  # 余弦衰减
                self.lr = target_lr + (base_lr - target_lr) * (1 + cosine) / 2  # 计算学习率
            elif lr_mode == 'linear':
                self.lr = target_lr + \
                          (base_lr - target_lr) * (1 - current_ratio)  # 线性衰减
            elif lr_mode == 'decay':
                epoch = step // self.training_config.steps_per_epoch  # 计算当前的 epoch
                self.lr = base_lr * lr_config.lr_decay ** epoch  # 计算学习率

        # 更新优化器中的每个参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr


def lr_scheduler_factory(lr_configs: List[DictConfig], cfg: DictConfig) -> List[LRScheduler]:
    """
    根据配置创建多个学习率调度器。

    输入：
    - lr_configs (List[DictConfig]): 包含每个优化器的学习率调度配置的列表。
    - cfg (DictConfig): 训练的总体配置。

    输出：
    - List[LRScheduler]: 创建的学习率调度器列表。
    """
    return [LRScheduler(cfg=cfg, optimizer_cfg=lr_config) for lr_config in lr_configs]

import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from omegaconf import DictConfig
from .base import BaseModel
"""
初始化 (__init__ 方法)
self.attention_list: 存储多个 TransformerEncoderLayer 实例。
self.readout: 决定如何聚合节点特征（"concat", "sum", "mean", "max"）。
self.node_num: 节点数量（cfg.dataset.node_sz）。
self.dim_reduction: 用于特征维度减少和连接池化的线性层，适用于 "concat" 聚合方式。
self.norm: 用于归一化的 Batch Normalization 层，适用于 "sum" 聚合方式。
self.fc: 最后的全连接层，将特征映射到最终的分类输出。
前向传播 (forward 方法)
输入:
node_feature 形状为 (batch_size, num_regions, node_feature_dim)。
处理:
通过多个 Transformer 编码器层变换节点特征。
根据 readout 类型进行特征聚合（例如，连接、求均值、最大值或求和）。
输出:
最终的分类结果，形状为 (batch_size, 2)，即两个类别的预测值。
获取注意力权重 (get_attention_weights 方法)
功能: 返回所有 Transformer 编码器层的注意力权重，用于分析模型的注意力机制。
"""

class GraphTransformer(BaseModel):
    '''
    图数据的 Transformer 模型。
    '''

    def __init__(self, cfg: DictConfig):
        '''
        初始化 GraphTransformer 实例。

        参数:
        cfg (DictConfig): 配置文件的 DictConfig 实例，包含模型的配置信息。
        '''
        super().__init__()

        self.attention_list = nn.ModuleList()
        self.readout = cfg.model.readout  # 读出方式 ('concat', 'sum', 'mean', 'max')
        self.node_num = cfg.dataset.node_sz  # 节点数量
        # 创建 Transformer 编码器层
        for _ in range(cfg.model.self_attention_layer):
            self.attention_list.append(
                TransformerEncoderLayer(
                    d_model=cfg.dataset.node_feature_sz,  # 输入特征维度
                    nhead=4,  # 注意力头数
                    dim_feedforward=1024,  # 前馈网络的维度
                    batch_first=True,  # 输入数据的 batch 维度在最前面
                    atten_para = None
                )
            )

        final_dim = cfg.dataset.node_feature_sz  # 初始特征维度

        # 根据 readout 类型选择不同的处理方式
        if self.readout == "concat":
            # 使用线性层进行维度减少，然后连接池化后的节点特征
            self.dim_reduction = nn.Sequential(
                nn.Linear(cfg.dataset.node_feature_sz, 8),  # 从原特征维度到 8
                nn.LeakyReLU()
            )
            final_dim = 8 * self.node_num  # 连接池化后的特征维度
        elif self.readout == "sum":
            # 对节点特征求和后进行 Batch Normalization
            self.norm = nn.BatchNorm1d(cfg.dataset.node_feature_sz)  # 对特征进行归一化
        # 最终的全连接层
        self.fc = nn.Sequential(
            nn.Linear(final_dim, 256),  # 从最终维度到 256
            nn.LeakyReLU(),
            nn.Linear(256, 32),  # 从 256 到 32
            nn.LeakyReLU(),
            nn.Linear(32, 2)  # 从 32 到 2 (分类输出)
        )

    def forward(self, time_seires, node_feature):
        '''
        前向传播操作。

        参数:
        time_seires (torch.Tensor): 时间序列输入，形状为 (batch_size, num_regions, time_series)
        node_feature (torch.Tensor): 节点特征，形状为 (batch_size, num_regions, node_feature_dim)

        返回:
        torch.Tensor: 模型输出，形状为 (batch_size, 2)
        '''
        bz, _, _ = node_feature.shape
        attn_param = node_feature


        # 通过每个 Transformer 编码器层
        for atten in self.attention_list:
            node_feature = atten(node_feature, attn_param)  # 维度变换后 (batch_size, num_regions, node_feature_dim)

        # 根据 readout 类型选择不同的池化操作
        if self.readout == "concat":
            node_feature = self.dim_reduction(node_feature)  # 形状为 (batch_size, num_regions, 8)
            node_feature = node_feature.reshape((bz, -1))  # 形状为 (batch_size, 8 * num_regions)

        elif self.readout == "mean":
            node_feature = torch.mean(node_feature, dim=1)  # 形状为 (batch_size, node_feature_dim)
        elif self.readout == "max":
            node_feature, _ = torch.max(node_feature, dim=1)  # 形状为 (batch_size, node_feature_dim)
        elif self.readout == "sum":
            node_feature = torch.sum(node_feature, dim=1)  # 形状为 (batch_size, node_feature_dim)
            node_feature = self.norm(node_feature)  # 形状为 (batch_size, node_feature_dim)

        return self.fc(node_feature)  # 形状为 (batch_size, 2)

    def get_attention_weights(self):
        '''
        获取每个自注意力层的权重。

        返回:
        List[torch.Tensor]: 包含每个自注意力层权重的列表。
        '''
        return [atten.get_attention_weights() for atten in self.attention_list]

import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from omegaconf import DictConfig
from .base import BaseModel


class GraphTransformer(BaseModel):

    def __init__(self, cfg: DictConfig):
        '''
        初始化 GraphTransformer 实例。

        参数:
        cfg (DictConfig): 配置文件的 DictConfig 实例，包含模型的配置信息。
        '''
        super().__init__()

        self.attention_list = nn.ModuleList()
        self.readout = cfg.model.readout  
        self.node_num = cfg.dataset.node_sz  # 节点数量

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

        if self.readout == "concat":

            self.dim_reduction = nn.Sequential(
                nn.Linear(cfg.dataset.node_feature_sz, 8),  # 从原特征维度到 8
                nn.LeakyReLU()
            )
            final_dim = 8 * self.node_num  # 连接池化后的特征维度
        elif self.readout == "sum":

            self.norm = nn.BatchNorm1d(cfg.dataset.node_feature_sz)  # 对特征进行归一化

        self.fc = nn.Sequential(
            nn.Linear(final_dim, 256),  # 从最终维度到 256
            nn.LeakyReLU(),
            nn.Linear(256, 32),  # 从 256 到 32
            nn.LeakyReLU(),
            nn.Linear(32, 2)  # 从 32 到 2 (分类输出)
        )

    def forward(self, time_seires, node_feature):

        bz, _, _ = node_feature.shape
        attn_param = node_feature


        for atten in self.attention_list:
            node_feature = atten(node_feature, attn_param)  # 维度变换后 (batch_size, num_regions, node_feature_dim)

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


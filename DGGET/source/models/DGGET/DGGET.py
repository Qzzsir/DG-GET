# import torch
# import torch.nn as nn
# from source.models.DGGET.ptdec import DEC
# from source.models.DGGET.components.Transformer_encoder import InterpretableTransformerEncoder
# from omegaconf import DictConfig
# from ..base import BaseModel
# import numpy as np
# from scipy.stats import entropy
# from source.dataset.domain_classifier import DomainClassifier
# from source.dataset.domain_classifier import GradientReversalLayer
#
# class TransPoolingEncoder(nn.Module):
#     """
#     Transformer encoder with Pooling mechanism.
#     Input size: (batch_size, input_node_num, input_feature_size)
#     Output size: (batch_size, output_node_num, input_feature_size)
#     """
#
#     def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True,
#                  freeze_center=False, project_assignment=True, atten_para_size=None):
#         super().__init__()
#         self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=4,
#                                                            dim_feedforward=hidden_size,
#                                                            batch_first=True
#                                                            )
#         if atten_para_size is not None:
#             self.atten_para = nn.Parameter(torch.zeros(input_node_num, atten_para_size))
#
#         self.pooling = pooling
#
#         if pooling:
#             encoder_hidden_size = 32
#             self.encoder = nn.Sequential(
#                 nn.Linear(input_feature_size * input_node_num, encoder_hidden_size),
#                 nn.LeakyReLU(),
#                 nn.Linear(encoder_hidden_size, encoder_hidden_size),
#                 nn.LeakyReLU(),
#                 nn.Linear(encoder_hidden_size, input_feature_size * input_node_num),
#             )
#             # ========== 关键修复3：使用 DEC 类而非模块 ==========
#             self.dec = DECClass(  # 用 DECClass（实际类）替代 DEC（模块）
#                 cluster_number=output_node_num,
#                 hidden_dimension=input_feature_size,
#                 encoder=self.encoder,
#                 orthogonal=orthogonal,
#                 freeze_center=freeze_center,
#                 project_assignment=project_assignment
#             )
#             # self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
#             #                orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)
#
#
#
#     def is_pooling_enabled(self):
#         return self.pooling
#
#     def forward(self, x, atten_para=None):
#
#         x = self.transformer(x, atten_para)
#
#         if self.pooling:
#             x, assignment = self.dec(x)
#             return x, assignment
#         return x, None
#
#     def get_attention_weights(self):
#         return self.transformer.get_attention_weights()
#
#     def loss(self, assignment):
#         return self.dec.loss(assignment)
#
#
# def dispersion_entropy(signal, m=2, c=6):
#     n = len(signal)
#     bins = np.percentile(signal, np.linspace(0, 100, c + 1))
#     digitized = np.digitize(signal, bins[:-1]) - 1
#     patterns = np.array([digitized[i:i + m] for i in range(n - m + 1)])
#     unique_patterns, counts = np.unique(patterns, axis=0, return_counts=True)
#     probabilities = counts / len(patterns)
#     disp_entropy = entropy(probabilities)
#     return disp_entropy
#
# def calculate_dispersion_entropy(data):
#
#     samples, channels, time_series = data.shape
#     disp_entropy_values = np.zeros((samples, channels))
#
#     for i in range(samples):
#         for j in range(channels):
#             disp_entropy_values[i, j] = dispersion_entropy(data[i, j])
#
#     return disp_entropy_values
#
#
#
# class BrainNetworkTransformer(BaseModel):
#     def __init__(self, config: DictConfig):
#         super().__init__()
#         self.atten_para = nn.Parameter(torch.zeros(config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
#         self.attention_list = nn.ModuleList()
#         forward_dim = config.dataset.node_sz
#
#         self.pos_encoding = config.model.pos_encoding
#         if self.pos_encoding == 'identity':
#             self.node_identity = nn.Parameter(torch.zeros(
#                 config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
#             forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
#             nn.init.kaiming_normal_(self.node_identity)
#
#         sizes = config.model.sizes
#         sizes[0] = config.dataset.node_sz
#         in_sizes = [config.dataset.node_sz] + sizes[:-1]
#         do_pooling = config.model.pooling
#         self.do_pooling = do_pooling
#         for index, size in enumerate(sizes):
#             self.attention_list.append(
#                 TransPoolingEncoder(input_feature_size=forward_dim,
#                                     input_node_num=in_sizes[index],
#                                     hidden_size=1024,
#                                     output_node_num=size,
#                                     pooling=do_pooling[index],
#                                     orthogonal=config.model.orthogonal,
#                                     freeze_center=config.model.freeze_center,
#                                     # atten_para=None,
#                                     project_assignment=config.model.project_assignment))
#
#         self.dim_reduction = nn.Sequential(
#             nn.Linear(forward_dim, 8),
#             nn.LeakyReLU()
#         )
#
#         self.fc = nn.Sequential(
#             nn.Linear(8 * sizes[-1], 256),
#             nn.LeakyReLU(),
#             nn.Linear(256, 32),
#             nn.LeakyReLU(),
#             nn.Linear(32, 2)
#         )
#
#         input_dim = 8 * sizes[-1]
#         hidden_dim = 128
#         self.domain_classifier = DomainClassifier(input_dim=input_dim, hidden_dim=hidden_dim)
#
#     def forward(self,
#                 time_series: torch.Tensor,
#                 node_feature: torch.Tensor,
#                 site: torch.Tensor,
#                 grl_lambda: None):
#
#         domain_labels = torch.argmax(site, dim=1)
#         bz, _, _, = node_feature.shape
#
#         if self.pos_encoding == 'identity':
#             pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
#             node_feature = torch.cat([node_feature, pos_emb], dim=-1)
#
#         assignments = []
#         disp_entropy_values = calculate_dispersion_entropy(time_series)
#         disp_entropy_values = torch.tensor(disp_entropy_values, dtype=torch.float32)
#
#         normalized_results = []
#
#         for i in range(disp_entropy_values.shape[0]):
#             vector = disp_entropy_values[i].squeeze()
#             diff_matrix = torch.abs(vector.view(-1, 1) - vector.view(1, -1))
#             sigma = diff_matrix.std() + 1e-10
#             similarity_matrix = torch.exp(-diff_matrix / sigma)
#
#             min_val = similarity_matrix.min()
#             max_val = similarity_matrix.max()
#             normalized = (similarity_matrix - min_val) / (max_val - min_val + 1e-10)
#
#             normalized_results.append(normalized)
#
#         normalized_results = torch.stack(normalized_results)
#         atten_para = normalized_results
#         for atten in self.attention_list:
#             node_feature, assignment = atten(node_feature, atten_para)
#             assignments.append(assignment)
#
#         node_feature = self.dim_reduction(node_feature)
#         node_feature = node_feature.reshape((bz, -1))
#
#         class_output = self.fc(node_feature)
#         if grl_lambda is not None :
#             lambda_reversal = grl_lambda
#         else:
#             lambda_reversal = 0
#         reversed_features = GradientReversalLayer.apply(
#             node_feature, lambda_reversal
#         )
#         domain_output = self.domain_classifier(reversed_features)
#         domain_loss = None
#         if domain_labels is not None:
#             domain_loss = self.calculate_domain_loss(domain_output, domain_labels)
#         return class_output, domain_output, domain_loss
#
#     def get_learnable_matrix(self, time_series, node_feature, site):
#         return self.learnable_matrix
#
#     def calculate_domain_loss(self, domain_output, domain_labels):
#         criterion = nn.CrossEntropyLoss()
#         return criterion(domain_output, domain_labels)
#
#
#
#     def get_attention_weights(self):
#         return [atten.get_attention_weights() for atten in self.attention_list]
#
#     def get_cluster_centers(self) -> torch.Tensor:
#         """
#         Get the cluster centers, as computed by the encoder.
#
#         :return: [number of clusters, hidden dimension] Tensor of dtype float
#         """
#         return self.dec.get_cluster_centers()
#
#     def loss(self, assignments):
#         """
#         Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
#         Inputs: assignments: [batch size, number of clusters]
#         Output: KL loss
#         """
#         decs = list(
#             filter(lambda x: x.is_pooling_enabled(), self.attention_list))
#         assignments = list(filter(lambda x: x is not None, assignments))
#         loss_all = None
#
#         for index, assignment in enumerate(assignments):
#             if loss_all is None:
#                 loss_all = decs[index].loss(assignment)
#             else:
#                 loss_all += decs[index].loss(assignment)
#         return loss_all
#
# # import torch
# # import torch.nn as nn
# # from omegaconf import DictConfig
# # import numpy as np
# # from scipy.stats import entropy
# #
# # # ========== 关键修复1：正确导入 DEC 类 ==========
# # # 假设 ptdec.py 中定义的 DEC 类名为 DEC，需确保导入的是类而非模块
# # # 如果 ptdec.py 中是 class DEC(...)，则导入方式：
# # from .ptdec import DEC as DECClass  # 重命名避免模块/类混淆
# # # 如果 DEC 类在 ptdec.py 的子模块中，需调整为：
# # # from .ptdec.dec import DEC as DECClass
# #
# # from source.models.DGGET.components.Transformer_encoder import InterpretableTransformerEncoder
# # from ..base import BaseModel
# # from source.dataset.domain_classifier import DomainClassifier, GradientReversalLayer
# #
# #
# # # ========== TransPoolingEncoder 完整修复 ==========
# # class TransPoolingEncoder(nn.Module):
# #     """
# #     Transformer encoder with Pooling mechanism.
# #     Input size: (batch_size, input_node_num, input_feature_size)
# #     Output size: (batch_size, output_node_num, input_feature_size)
# #     """
# #
# #     def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num,
# #                  pooling=True, orthogonal=True, freeze_center=False, project_assignment=True,
# #                  atten_para_size=None, atten_para=None):  # 保留 atten_para 参数
# #         super().__init__()
# #
# #         # ========== 关键修复2：将 atten_para 传入 Transformer ==========
# #         self.transformer = InterpretableTransformerEncoder(
# #             d_model=input_feature_size,
# #             nhead=4,
# #             dim_feedforward=hidden_size,
# #             batch_first=True,
# #             atten_para=atten_para  # 启用 atten_para 参数传递
# #         )
# #
# #         # 初始化可学习的 atten_para（如果指定了尺寸）
# #         if atten_para_size is not None:
# #             self.atten_para = nn.Parameter(torch.zeros(input_node_num, atten_para_size))
# #
# #         self.pooling = pooling
# #
# #         if pooling:
# #             encoder_hidden_size = 32
# #             self.encoder = nn.Sequential(
# #                 nn.Linear(input_feature_size * input_node_num, encoder_hidden_size),
# #                 nn.LeakyReLU(),
# #                 nn.Linear(encoder_hidden_size, encoder_hidden_size),
# #                 nn.LeakyReLU(),
# #                 nn.Linear(encoder_hidden_size, input_feature_size * input_node_num),
# #             )
# #
# #             # ========== 关键修复3：使用 DEC 类而非模块 ==========
# #             self.dec = DECClass(  # 用 DECClass（实际类）替代 DEC（模块）
# #                 cluster_number=output_node_num,
# #                 hidden_dimension=input_feature_size,
# #                 encoder=self.encoder,
# #                 orthogonal=orthogonal,
# #                 freeze_center=freeze_center,
# #                 project_assignment=project_assignment
# #             )
# #
# #     def is_pooling_enabled(self):
# #         return self.pooling
# #
# #     def forward(self, x, atten_para=None):
# #         # 前向传播时优先使用传入的 atten_para（色散熵计算的矩阵）
# #         x = self.transformer(x, atten_para)
# #
# #         if self.pooling:
# #             x, assignment = self.dec(x)
# #             return x, assignment
# #         return x, None
# #
# #     def get_attention_weights(self):
# #         return self.transformer.get_attention_weights()
# #
# #     def loss(self, assignment):
# #         return self.dec.loss(assignment)
# #
# #
# # # ========== 色散熵计算函数（补充完整） ==========
# # def calculate_dispersion_entropy_for_eeg(time_series):
# #     """适配EEG数据的色散熵计算（补全你缺失的实现）"""
# #
# #     def dispersion_entropy(signal, m=2, c=6):
# #         n = len(signal)
# #         bins = np.percentile(signal, np.linspace(0, 100, c + 1))
# #         digitized = np.digitize(signal, bins[:-1]) - 1
# #         patterns = np.array([digitized[i:i + m] for i in range(n - m + 1)])
# #         unique_patterns, counts = np.unique(patterns, axis=0, return_counts=True)
# #         probabilities = counts / len(patterns)
# #         return entropy(probabilities)
# #
# #     # 适配批量数据：(batch_size, num_regions, time_points)
# #     samples, channels, time_series_len = time_series.shape
# #     disp_entropy_values = np.zeros((samples, channels))
# #
# #     for i in range(samples):
# #         for j in range(channels):
# #             disp_entropy_values[i, j] = dispersion_entropy(time_series[i, j].cpu().numpy())
# #
# #     return disp_entropy_values
# #
# #
# # # ========== BrainNetworkTransformer 完整修复 ==========
# # class BrainNetworkTransformer(BaseModel):
# #     def __init__(self, config: DictConfig):
# #         super().__init__()
# #
# #         # 可学习的注意力参数
# #         self.atten_para = nn.Parameter(
# #             torch.zeros(config.dataset.node_sz, config.model.pos_embed_dim),
# #             requires_grad=True
# #         )
# #         self.attention_list = nn.ModuleList()
# #         forward_dim = config.dataset.node_sz
# #
# #         # 位置编码
# #         self.pos_encoding = config.model.pos_encoding
# #         if self.pos_encoding == 'identity':
# #             self.node_identity = nn.Parameter(
# #                 torch.zeros(config.dataset.node_sz, config.model.pos_embed_dim),
# #                 requires_grad=True
# #             )
# #             forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
# #             nn.init.kaiming_normal_(self.node_identity)
# #
# #         # 模型层配置
# #         sizes = config.model.sizes
# #         sizes[0] = config.dataset.node_sz
# #         in_sizes = [config.dataset.node_sz] + sizes[:-1]
# #         do_pooling = config.model.pooling
# #         self.do_pooling = do_pooling
# #
# #         # ========== 保留 atten_para 参数传入 TransPoolingEncoder ==========
# #         for index, size in enumerate(sizes):
# #             self.attention_list.append(
# #                 TransPoolingEncoder(
# #                     input_feature_size=forward_dim,
# #                     input_node_num=in_sizes[index],
# #                     hidden_size=1024,
# #                     output_node_num=size,
# #                     pooling=do_pooling[index],
# #                     orthogonal=config.model.orthogonal,
# #                     freeze_center=config.model.freeze_center,
# #                     atten_para=None,  # 初始化时不传，前向传播时传入计算好的矩阵
# #                     project_assignment=config.model.project_assignment
# #                 )
# #             )
# #
# #         # 维度降维层
# #         self.dim_reduction = nn.Sequential(
# #             nn.Linear(forward_dim, 8),
# #             nn.LeakyReLU()
# #         )
# #
# #         # 分类头
# #         self.fc = nn.Sequential(
# #             nn.Linear(8 * sizes[-1], 256),
# #             nn.LeakyReLU(),
# #             nn.Linear(256, 32),
# #             nn.LeakyReLU(),
# #             nn.Linear(32, 2)
# #         )
# #
# #         # 域分类器
# #         input_dim = 8 * sizes[-1]
# #         hidden_dim = 128
# #         self.domain_classifier = DomainClassifier(input_dim=input_dim, hidden_dim=hidden_dim)
# #
# #     # 排列熵计算（保留）
# #     def multi_channel_permutation_entropy(self, time_series: torch.Tensor, m: int, delay: int = 1) -> torch.Tensor:
# #         """占位：需补全排列熵实现"""
# #         return torch.zeros(time_series.shape[0], time_series.shape[1])
# #
# #     def get_lambda(self, step, total_steps):
# #         """Progressive GRL scheduling (DANN-style)"""
# #         p = step / total_steps
# #         return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
# #
# #     def forward(self, time_series: torch.Tensor, node_feature: torch.Tensor,
# #                 site: torch.Tensor, step: int = None, total_steps: int = None):
# #         domain_labels = torch.argmax(site, dim=1)
# #         bz, _, _ = node_feature.shape
# #
# #         # 位置编码拼接
# #         if self.pos_encoding == 'identity':
# #             pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
# #             node_feature = torch.cat([node_feature, pos_emb], dim=-1)
# #
# #         # ========== 核心：计算色散熵并生成 atten_para ==========
# #         disp_entropy_values = calculate_dispersion_entropy_for_eeg(time_series)
# #         disp_entropy_values = torch.tensor(disp_entropy_values, dtype=torch.float32, device=node_feature.device)
# #
# #         # 生成归一化的注意力矩阵（atten_para）
# #         normalized_results = []
# #         for i in range(disp_entropy_values.shape[0]):
# #             vector = disp_entropy_values[i].squeeze()
# #             vector_inv = vector.view(-1, 1)
# #             vector_inv_transpose = vector_inv.T
# #             product = vector_inv @ vector_inv_transpose
# #
# #             # Min-Max 归一化
# #             min_val = product.min()
# #             max_val = product.max()
# #             normalized = (product - min_val) / (max_val - min_val + 1e-10)
# #             normalized_results.append(normalized)
# #
# #         # 最终 atten_para：(batch_size, node_num, node_num)
# #         atten_para = torch.stack(normalized_results)
# #
# #         # ========== 前向传播：将 atten_para 传入每个 TransPoolingEncoder ==========
# #         assignments = []
# #         for atten in self.attention_list:
# #             node_feature, assignment = atten(node_feature, atten_para)  # 传入计算好的 atten_para
# #             assignments.append(assignment)
# #
# #         # 维度降维 + 展平
# #         node_feature = self.dim_reduction(node_feature)
# #         node_feature = node_feature.reshape((bz, -1))
# #
# #         # 分类输出
# #         class_output = self.fc(node_feature)
# #
# #         # 域适应（GRL）
# #         if step is not None and total_steps is not None:
# #             lambda_reversal = self.get_lambda(step, total_steps)
# #         else:
# #             lambda_reversal = 1.0
# #
# #         reversed_features = GradientReversalLayer.apply(node_feature, lambda_reversal)
# #         domain_output = self.domain_classifier(reversed_features)
# #
# #         # 计算域损失
# #         domain_loss = None
# #         if domain_labels is not None:
# #             criterion = nn.CrossEntropyLoss()
# #             domain_loss = criterion(domain_output, domain_labels)
# #
# #         return class_output, domain_output, domain_loss
# #
# #     # 辅助方法（保留）
# #     def get_learnable_matrix(self, time_series, node_feature, site):
# #         return getattr(self, 'learnable_matrix', None)  # 避免未定义报错
# #
# #     def calculate_domain_loss(self, domain_output, domain_labels):
# #         criterion = nn.CrossEntropyLoss()
# #         return criterion(domain_output, domain_labels)
# #
# #     def get_attention_weights(self):
# #         return [atten.get_attention_weights() for atten in self.attention_list]
# #
# #     def get_cluster_centers(self) -> torch.Tensor:
# #         """修复：遍历找到第一个有 DEC 的层，返回聚类中心"""
# #         for atten in self.attention_list:
# #             if atten.pooling and hasattr(atten, 'dec'):
# #                 return atten.dec.get_cluster_centers()
# #         return torch.tensor([])
# #
# #     def loss(self, assignments):
# #         decs = list(filter(lambda x: x.is_pooling_enabled(), self.attention_list))
# #         assignments = list(filter(lambda x: x is not None, assignments))
# #         loss_all = None
# #
# #         for index, assignment in enumerate(assignments):
# #             if loss_all is None:
# #                 loss_all = decs[index].loss(assignment)
# #             else:
# #                 loss_all += decs[index].loss(assignment)
# #         return loss_all if loss_all is not None else torch.tensor(0.0)


import torch
import torch.nn as nn
from omegaconf import DictConfig
import numpy as np
from scipy.stats import entropy

from source.models.DGGET.ptdec.dec import DEC
# 如果 DEC 类在 ptdec/dec.py 中，改为：from .ptdec.dec import DEC

from source.models.DGGET.components.Transformer_encoder import InterpretableTransformerEncoder
from ..base import BaseModel
from source.dataset.domain_classifier import DomainClassifier, GradientReversalLayer


# ===================== TransPoolingEncoder 完整修正 =====================
class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """
    # 这是一个带有池化机制的 Transformer 编码器类，继承自 nn.Module

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True,
                 freeze_center=False, project_assignment=True, atten_para_size=None, atten_para=None):
        super().__init__()
        # 初始化方法，设置编码器的输入和输出大小，以及是否启用池化等参数
        # print("input_feature_size",input_feature_size)


        self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=4,
                                                           dim_feedforward=hidden_size,
                                                           batch_first=True
                                                           # , atten_para=atten_para
                                                           )
        # 创建一个可解释的 Transformer 编码器实例
        # 输入: (batch_size, input_node_num, input_feature_size)
        # 输出: (batch_size, input_node_num, input_feature_size)

        # 初始化 atten_para
        if atten_para_size is not None:
            self.atten_para = nn.Parameter(torch.zeros(input_node_num, atten_para_size))  # 初始化为零

        self.pooling = pooling
        # 记录是否启用池化的标志

        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size * input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, input_feature_size * input_node_num),
            )
            # 如果启用了池化，创建一个全连接网络作为编码器
            # 输入: (batch_size, input_node_num, input_feature_size)
            # 输出: (batch_size, input_node_num, input_feature_size)

            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)
            # 创建 DEC 模块实例，用于聚类
            # 输入: (batch_size, input_node_num, input_feature_size)
            # 输出: (batch_size, output_node_num, input_feature_size) 和 (batch_size, output_node_num)

    def is_pooling_enabled(self):
        return self.pooling
        # 返回池化是否启用的标志

    def forward(self, x, atten_para=None):

        x = self.transformer(x, atten_para)


        # 通过 Transformer 编码器处理输入数据
        # 输入: (batch_size, input_node_num, input_feature_size)
        # 输出: (batch_size, input_node_num, input_feature_size)

        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None
        # 如果启用了池化，将输出传递给 DEC 进行池化，返回池化结果和分配矩阵
        # 输出: (batch_size, output_node_num, input_feature_size) 和 (batch_size, output_node_num) 或 (batch_size, input_node_num, input_feature_size) 和 None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()
        # 获取 Transformer 编码器的注意力权重

    def loss(self, assignment):
        return self.dec.loss(assignment)
        # 计算 DEC 模块的损失值



# ===================== 色散熵计算（补全，适配你的数据） =====================
def calculate_dispersion_entropy_for_eeg(time_series):
    def dispersion_entropy(signal, m=2, c=6):
        n = len(signal)
        bins = np.percentile(signal, np.linspace(0, 100, c + 1))
        digitized = np.digitize(signal, bins[:-1]) - 1
        patterns = np.array([digitized[i:i + m] for i in range(n - m + 1)])
        unique_patterns, counts = np.unique(patterns, axis=0, return_counts=True)
        probabilities = counts / len(patterns)
        return entropy(probabilities)

    # 适配批量 EEG 数据：(batch_size, num_regions, time_points)
    samples, channels, time_series_len = time_series.shape
    disp_entropy_values = np.zeros((samples, channels))

    # 兼容 GPU/CPU 数据
    ts_np = time_series.cpu().numpy() if torch.is_tensor(time_series) else time_series
    for i in range(samples):
        for j in range(channels):
            disp_entropy_values[i, j] = dispersion_entropy(ts_np[i, j])

    return disp_entropy_values


# ===================== BrainNetworkTransformer 完整修正 =====================
class BrainNetworkTransformer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__()

        # 可学习的 atten_para 参数（你的定义）
        self.atten_para = nn.Parameter(
            torch.zeros(config.dataset.node_sz, config.model.pos_embed_dim),
            requires_grad=True
        )
        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        # 位置编码
        self.pos_encoding = config.model.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(
                torch.zeros(config.dataset.node_sz, config.model.pos_embed_dim),
                requires_grad=True
            )
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        # 模型层配置
        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling

        # ===================== 保留你的 atten_para 参数传入 =====================
        for index, size in enumerate(sizes):
            self.attention_list.append(
                TransPoolingEncoder(
                    input_feature_size=forward_dim,
                    input_node_num=in_sizes[index],
                    hidden_size=1024,
                    output_node_num=size,
                    pooling=do_pooling[index],
                    orthogonal=config.model.orthogonal,
                    freeze_center=config.model.freeze_center,
                    atten_para=None,  # 初始化不传，前向传播时传计算好的矩阵
                    project_assignment=config.model.project_assignment
                )
            )

        # 维度降维层
        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        # 分类头
        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

        # 域分类器
        input_dim = 8 * sizes[-1]
        hidden_dim = 128
        self.domain_classifier = DomainClassifier(input_dim=input_dim, hidden_dim=hidden_dim)

    # 排列熵计算（占位，不影响核心运行）
    def multi_channel_permutation_entropy(self, time_series: torch.Tensor, m: int, delay: int = 1) -> torch.Tensor:
        return torch.zeros(time_series.shape[0], time_series.shape[1], device=time_series.device)

    def get_lambda(self, step, total_steps):
        """Progressive GRL scheduling (DANN-style)"""
        p = step / total_steps
        return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

    def forward(self, time_series: torch.Tensor, node_feature: torch.Tensor,
                site: torch.Tensor, step: int = None, total_steps: int = None):
        domain_labels = torch.argmax(site, dim=1)
        bz, _, _ = node_feature.shape

        # 位置编码拼接
        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        # ===================== 你的核心逻辑：计算色散熵生成 atten_para =====================
        disp_entropy_values = calculate_dispersion_entropy_for_eeg(time_series)
        disp_entropy_values = torch.tensor(disp_entropy_values, dtype=torch.float32, device=node_feature.device)

        # 生成归一化的注意力矩阵（atten_para）
        normalized_results = []
        for i in range(disp_entropy_values.shape[0]):
            vector = disp_entropy_values[i].squeeze()
            vector_inv = vector.view(-1, 1)
            vector_inv_transpose = vector_inv.T
            product = vector_inv @ vector_inv_transpose

            # Min-Max 归一化（防止除零）
            min_val = product.min()
            max_val = product.max()
            normalized = (product - min_val) / (max_val - min_val + 1e-10)
            normalized_results.append(normalized)

        # 最终 atten_para：(batch_size, node_num, node_num) —— 传入 Transformer
        atten_para = torch.stack(normalized_results)

        # ===================== 前向传播：将 atten_para 传入每个 TransPoolingEncoder =====================
        assignments = []
        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature, atten_para)  # 传入你的 atten_para
            assignments.append(assignment)

        # 维度降维 + 展平
        node_feature = self.dim_reduction(node_feature)
        node_feature = node_feature.reshape((bz, -1))

        # 分类输出
        class_output = self.fc(node_feature)

        # 域适应（GRL）
        if step is not None and total_steps is not None:
            lambda_reversal = self.get_lambda(step, total_steps)
        else:
            lambda_reversal = 1.0

        reversed_features = GradientReversalLayer.apply(node_feature, lambda_reversal)
        domain_output = self.domain_classifier(reversed_features)

        # 计算域损失
        domain_loss = None
        if domain_labels is not None:
            criterion = nn.CrossEntropyLoss()
            domain_loss = criterion(domain_output, domain_labels)

        return class_output, domain_output, domain_loss

    # ===================== 辅助方法（修正潜在报错） =====================
    def get_learnable_matrix(self, time_series, node_feature, site):
        # 避免未定义 learnable_matrix 报错
        return getattr(self, 'learnable_matrix', torch.tensor([]))

    def calculate_domain_loss(self, domain_output, domain_labels):
        criterion = nn.CrossEntropyLoss()
        return criterion(domain_output, domain_labels)

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """修正：遍历找到有 DEC 的层，返回聚类中心"""
        for atten in self.attention_list:
            if atten.pooling and hasattr(atten, 'dec'):
                return atten.dec.get_cluster_centers()
        return torch.tensor([])

    def loss(self, assignments):
        """修正：空列表保护"""
        decs = list(filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all if loss_all is not None else torch.tensor(0.0)
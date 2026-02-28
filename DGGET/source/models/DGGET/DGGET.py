import torch
import torch.nn as nn
from omegaconf import DictConfig
import numpy as np
from scipy.stats import entropy

from source.models.DGGET.ptdec.dec import DEC

from source.models.DGGET.components.Transformer_encoder import InterpretableTransformerEncoder
from ..base import BaseModel
from source.dataset.domain_classifier import DomainClassifier, GradientReversalLayer


class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True,
                 freeze_center=False, project_assignment=True, atten_para_size=None, atten_para=None):
        super().__init__()

        
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

            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)


    def is_pooling_enabled(self):
        return self.pooling
        # 返回池化是否启用的标志

    def forward(self, x, atten_para=None):

        x = self.transformer(x, atten_para)

        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


def calculate_dispersion_entropy_for_eeg(time_series):
    def dispersion_entropy(signal, m=2, c=6):
        n = len(signal)
        bins = np.percentile(signal, np.linspace(0, 100, c + 1))
        digitized = np.digitize(signal, bins[:-1]) - 1
        patterns = np.array([digitized[i:i + m] for i in range(n - m + 1)])
        unique_patterns, counts = np.unique(patterns, axis=0, return_counts=True)
        probabilities = counts / len(patterns)
        return entropy(probabilities)

    samples, channels, time_series_len = time_series.shape
    disp_entropy_values = np.zeros((samples, channels))

    ts_np = time_series.cpu().numpy() if torch.is_tensor(time_series) else time_series
    for i in range(samples):
        for j in range(channels):
            disp_entropy_values[i, j] = dispersion_entropy(ts_np[i, j])

    return disp_entropy_values


# ===================== BrainNetworkTransformer =====================
class BrainNetworkTransformer(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__()

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

        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling

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

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

        input_dim = 8 * sizes[-1]
        hidden_dim = 128
        self.domain_classifier = DomainClassifier(input_dim=input_dim, hidden_dim=hidden_dim)

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

        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        disp_entropy_values = calculate_dispersion_entropy_for_eeg(time_series)
        disp_entropy_values = torch.tensor(disp_entropy_values, dtype=torch.float32, device=node_feature.device)

        normalized_results = []
        for i in range(disp_entropy_values.shape[0]):
            vector = disp_entropy_values[i].squeeze()
            vector_inv = vector.view(-1, 1)
            vector_inv_transpose = vector_inv.T
            product = vector_inv @ vector_inv_transpose

            min_val = product.min()
            max_val = product.max()
            normalized = (product - min_val) / (max_val - min_val + 1e-10)
            normalized_results.append(normalized)

        atten_para = torch.stack(normalized_results)

        assignments = []
        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature, atten_para)  # 传入你的 atten_para
            assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)
        node_feature = node_feature.reshape((bz, -1))

        class_output = self.fc(node_feature)

        if step is not None and total_steps is not None:
            lambda_reversal = self.get_lambda(step, total_steps)
        else:
            lambda_reversal = 1.0

        reversed_features = GradientReversalLayer.apply(node_feature, lambda_reversal)
        domain_output = self.domain_classifier(reversed_features)

        domain_loss = None
        if domain_labels is not None:
            criterion = nn.CrossEntropyLoss()
            domain_loss = criterion(domain_output, domain_labels)

        return class_output, domain_output, domain_loss

    def get_learnable_matrix(self, time_series, node_feature, site):
        
        return getattr(self, 'learnable_matrix', torch.tensor([]))

    def calculate_domain_loss(self, domain_output, domain_labels):
        criterion = nn.CrossEntropyLoss()
        return criterion(domain_output, domain_labels)

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        for atten in self.attention_list:
            if atten.pooling and hasattr(atten, 'dec'):
                return atten.dec.get_cluster_centers()
        return torch.tensor([])

    def loss(self, assignments):
        decs = list(filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)

        return loss_all if loss_all is not None else torch.tensor(0.0)

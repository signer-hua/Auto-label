"""
MRF 多参考融合器（Multi-Reference Fuser）
基于纯 PyTorch 实现的多参考图特征融合模块。

核心组件：
    - 通道加权 MLP：学习特征通道的重要性权重
    - 自注意力模块：跨参考图特征交互
    - 低层/高层特征融合：DINOv3 层9（低层语义）+ 层12（高层抽象）加权合并

所有计算仅依赖 torch/numpy，适配半精度推理。
"""
import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ChannelWeightMLP(nn.Module):
    """通道加权 MLP：对特征通道进行重要性建模"""

    def __init__(self, embed_dim: int = 384, reduction: int = 4):
        super().__init__()
        mid_dim = max(embed_dim // reduction, 32)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C] 特征向量
        Returns:
            weighted: [B, C] 通道加权后的特征
        """
        weights = self.mlp(x)
        return x * weights


class MultiReferFuser(nn.Module):
    """
    多参考图特征融合器。

    融合策略：
    1. 对每个参考图的低层/高层特征用通道 MLP 加权
    2. 多参考图特征通过自注意力交互
    3. 输出 L2 归一化的融合特征向量
    """

    def __init__(self, embed_dim: int = 384, num_heads: int = 6):
        super().__init__()
        self.embed_dim = embed_dim

        self.low_channel_mlp = ChannelWeightMLP(embed_dim)
        self.high_channel_mlp = ChannelWeightMLP(embed_dim)

        self.layer_fuse_weight = nn.Parameter(torch.tensor([0.4, 0.6]))

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def fuse_features(
        self,
        low_features: list[np.ndarray],
        high_features: list[np.ndarray],
        weights: list[float] | None = None,
        device: str = "cuda",
    ) -> np.ndarray:
        """
        融合多参考图的低层+高层特征。

        Args:
            low_features: 各参考图的低层（层9）特征列表，每个 shape [C]
            high_features: 各参考图的高层（层12）特征列表，每个 shape [C]
            weights: 各参考图权重，None 则均分
            device: 计算设备

        Returns:
            fused: L2 归一化的融合特征，shape [C]
        """
        n = len(low_features)
        if n == 0:
            raise ValueError("No reference features to fuse")

        self.eval()
        self.to(device).half()

        low_t = torch.from_numpy(np.stack(low_features)).to(device).half()
        high_t = torch.from_numpy(np.stack(high_features)).to(device).half()

        low_weighted = self.low_channel_mlp(low_t)
        high_weighted = self.high_channel_mlp(high_t)

        lw = torch.softmax(self.layer_fuse_weight, dim=0)
        fused_layers = lw[0] * low_weighted + lw[1] * high_weighted

        if n > 1:
            seq = fused_layers.unsqueeze(0)
            attn_out, _ = self.self_attn(seq, seq, seq)
            attn_out = self.norm(attn_out + seq)

            if weights is not None:
                w = torch.tensor(weights, device=device, dtype=torch.float16)
                w = w / w.sum()
                w = w.unsqueeze(0).unsqueeze(-1)
                result = (attn_out * w).sum(dim=1).squeeze(0)
            else:
                result = attn_out.mean(dim=1).squeeze(0)
        else:
            result = fused_layers.squeeze(0)

        result = result.float().cpu().numpy()
        result = result / (np.linalg.norm(result) + 1e-8)

        logger.info("[MRF] Fused %d reference features (low+high layers)", n)
        return result


_mrf_instance: MultiReferFuser | None = None


def get_mrf_instance(embed_dim: int = 384) -> MultiReferFuser:
    """获取 MRF 单例"""
    global _mrf_instance
    if _mrf_instance is None:
        _mrf_instance = MultiReferFuser(embed_dim=embed_dim)
    return _mrf_instance

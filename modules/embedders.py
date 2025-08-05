from typing import Optional
import torch
import math
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, n_ch, frequency_embedding_size = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, n_ch, bias = True),
            nn.SiLU(),
            nn.Linear(n_ch, n_ch, bias = True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period = 10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start = 0, end = half, dtype = torch.float32) / half
        ).to(device = t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim = -1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim = -1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, emb_dim: int, drop_prob: float = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.drop_prob = drop_prob
        self.null_emb = nn.Parameter(torch.randn(emb_dim))
        self.norm = nn.LayerNorm(emb_dim)

    def _apply_cfg_dropout(self, labels: torch.Tensor, drop_ids: Optional[torch.Tensor] = None):
        '''
        训练时将一些标签有意地置为无效值，以使模型具备无条件生成能力。
        '''
        batch_size = labels.shape[0]
        if drop_ids is not None:
            drop_mask = drop_ids
        elif self.training and self.drop_prob > 0:
            drop_mask = torch.rand(batch_size, device = labels.device) < self.drop_prob
        else:
            return labels

        null_embed = self.null_emb.unsqueeze(0).expand(batch_size, -1)
        labels = torch.where(drop_mask.unsqueeze(-1), null_embed, labels)

        return labels


class CategoricalLabelEmbedder(LabelEmbedder):
    def __init__(self, cls_num: int, lab_emb_dim: int, drop_lab: float = 0.1):
        super().__init__(lab_emb_dim, drop_lab)
        self.cls_num = cls_num
        self.embedding = nn.Embedding(cls_num, lab_emb_dim)

        nn.init.normal_(self.embedding.weight, std = 0.02)
        nn.init.normal_(self.null_emb, std = 0.02)

    def forward(self, labels: Optional[torch.Tensor], use_drop: bool, force_drop_ids: Optional[torch.Tensor] = None):
        if labels is None:
            # Unconditional generation - use null embedding
            batch_size = force_drop_ids.shape[0] if force_drop_ids is not None else 1
            return self.null_emb.unsqueeze(0).expand(batch_size, -1)

        labels = torch.clamp(labels, 0, self.cls_num - 1)
        embeddings = self.embedding(labels)
        embeddings = self.norm(embeddings)
        if use_drop:
            embeddings = self._apply_cfg_dropout(embeddings, force_drop_ids)

        return embeddings

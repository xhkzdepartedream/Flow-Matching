import math
from typing import Union, List, Tuple, Optional

import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TimeEmbedding(nn.Module):
    def __init__(self, ch_num: int):
        super().__init__()
        self.ch_num = ch_num
        self.dense1 = nn.Linear(ch_num // 4, ch_num)
        self.dense2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(ch_num, ch_num)
        )

    def forward(self, t: torch.Tensor):  # t.shape = (batch_size,)
        emb_dim = self.ch_num // 8
        emb = math.log(10000) / (emb_dim - 1)
        emb = torch.exp(torch.arange(emb_dim, device = t.device) * (-emb))
        emb = t.reshape(-1, 1) @ emb.reshape(1, -1)  # shape = (batch_size, half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim = 1)  # shape = (batch_size, 2*half_dim)
        emb = self.dense1(emb)  # shape = (batch_size, ch_num)
        emb = self.dense2(emb)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int, n_groups: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.norm1 = nn.GroupNorm(n_groups, in_ch)
        self.norm2 = nn.GroupNorm(n_groups, out_ch)
        self.time_emb = nn.Linear(time_ch, out_ch)
        self.act1 = nn.SiLU()
        self.act2 = nn.SiLU()
        self.act3 = nn.SiLU()
        if out_ch == in_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

    def forward(self, input: torch.Tensor, t: torch.Tensor):
        output = self.conv1(self.act1(self.norm1(input)))
        tmp = self.time_emb(self.act2(t))[:, :, None, None]
        output += tmp

        output = self.conv2(self.act3(self.norm2(output)))
        output += self.shortcut(input)
        return output


class AttentionBlock(nn.Module):
    def __init__(self, n_ch: int, n_heads: int = 8, d_k: int = None, n_groups: int = 16):
        super().__init__()
        if d_k is None:
            d_k = n_ch // n_heads  # 更合理的默认值
        self.norm = nn.GroupNorm(n_groups, n_ch)
        self.w_qkv = nn.Linear(n_ch, n_heads * d_k * 3)
        self.dense = nn.Linear(n_heads * d_k, n_ch)
        self.scale = d_k ** (-0.5)
        self.d_k = d_k
        self.n_heads = n_heads

    def forward(self, input: torch.Tensor, t: Optional[torch.Tensor] = None):
        _ = t
        batch_size, n_ch, height, width = input.shape
        input = self.norm(input)  # 先进行归一化
        input = input.reshape(batch_size, n_ch, -1).permute(0, 2, 1)  # shape: [batch_size, index(h*w), n_channels]
        qkv = self.w_qkv(input).reshape(batch_size, -1, self.n_heads, self.d_k * 3)  # n_channels = d_k * n_heads
        q, k, v = torch.chunk(qkv, 3, dim = -1)  # shape: [batch_size, index, n_heads, d_k]
        q_k_product = torch.matmul(q, k.transpose(-2, -1))
        attn = q_k_product / self.scale
        attn = attn.softmax(dim = 2)
        res = torch.matmul(attn, v)
        res = res.reshape(batch_size, -1, self.n_heads * self.d_k)
        res = self.dense(res) + input  # 残差连接
        res = res.permute(0, 2, 1).reshape(batch_size, n_ch, height, width)
        return res


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int, has_attn: bool, 
                 attn_heads: int = 8, attn_d_k: int = None, n_groups: int = 16):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, time_ch, n_groups)
        if has_attn:
            self.attn = AttentionBlock(out_ch, attn_heads, attn_d_k, n_groups)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int, has_attn: bool,
                 attn_heads: int = 8, attn_d_k: int = None, n_groups: int = 16):
        super().__init__()
        self.res = ResidualBlock(in_ch + out_ch, out_ch, time_ch, n_groups)
        if has_attn:
            self.attn = AttentionBlock(out_ch, attn_heads, attn_d_k, n_groups)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_ch: int, time_ch: int, attn_heads: int = 8, attn_d_k: int = None, n_groups: int = 16):
        super().__init__()
        self.res1 = ResidualBlock(n_ch, n_ch, time_ch, n_groups)
        self.attn = AttentionBlock(n_ch, attn_heads, attn_d_k, n_groups)
        self.res2 = ResidualBlock(n_ch, n_ch, time_ch, n_groups)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class UpSample(nn.Module):
    def __init__(self, n_ch: int):
        super().__init__()
        self.convT = nn.ConvTranspose2d(n_ch, n_ch, 4, 2, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        x = self.convT(x)
        return x


class DownSample(nn.Module):
    def __init__(self, n_ch: int):
        super().__init__()
        self.convT = nn.Conv2d(n_ch, n_ch, 3, 2, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        x = self.convT(x)
        return x


class Unet(nn.Module):
    def __init__(self,
                 input_ch: int,
                 n_ch: int,
                 ch_mults: Union[Tuple[int, ...], List[int]],
                 is_attn: Union[Tuple[bool, ...], List[bool]],
                 n_blocks: int = 2,
                 attn_heads: int = 8,
                 attn_d_k: int = None,
                 n_groups: int = 16):
        """
        UNet模型
        
        Args:
            input_ch: 输入通道数
            n_ch: 基础通道数
            ch_mults: 通道乘数列表
            is_attn: 每一层是否使用注意力机制
            n_blocks: 每一层的残差块数量
            attn_heads: 注意力头数
            attn_d_k: 每个注意力头的维度
            n_groups: GroupNorm的组数
        """
        super().__init__()
        n = len(ch_mults)
        
        self.time_emb = TimeEmbedding(n_ch * 4)
        self.preprocessing = nn.Conv2d(input_ch, n_ch, 3, 1, 1)
        
        # 下采样路径
        down = []
        in_ch = out_ch = n_ch
        for i in range(n):
            out_ch = in_ch * ch_mults[i]
            for j in range(n_blocks):
                down.append(DownBlock(in_ch, out_ch, n_ch * 4, is_attn[i], 
                                    attn_heads, attn_d_k, n_groups))
                in_ch = out_ch

            if i < n - 1:
                down.append(DownSample(out_ch))
        self.down = nn.ModuleList(down)

        # 中间块
        self.middle = MiddleBlock(out_ch, n_ch * 4, attn_heads, attn_d_k, n_groups)

        # 上采样路径
        up = []
        for i in reversed(range(n)):
            out_ch = in_ch
            for j in range(n_blocks):
                up.append(UpBlock(in_ch, out_ch, n_ch * 4, is_attn[i],
                                attn_heads, attn_d_k, n_groups))
            out_ch = out_ch // ch_mults[i]
            up.append(UpBlock(in_ch, out_ch, n_ch * 4, is_attn[i],
                            attn_heads, attn_d_k, n_groups))
            in_ch = out_ch
            if i > 0:
                up.append(UpSample(out_ch))

        self.up = nn.ModuleList(up)
        self.norm = nn.Sequential(
            nn.GroupNorm(min(8, n_ch), n_ch),  # 适配较小通道数
            nn.SiLU()
        )
        self.final = nn.Conv2d(n_ch, input_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: Optional[torch.Tensor] = None):
        _ = c
        t = t.to(torch.float)
        t = self.time_emb(t)
        x = self.preprocessing(x)
        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)
        x = self.middle(x, t)
        for m in self.up:
            if isinstance(m, UpSample):
                x = m(x, t)
            else:
                s = h.pop()
                x = m(torch.cat((x, s), dim = 1), t)
        x = self.final(self.norm(x))
        return x


if __name__ == "__main__":
    # 测试UNet模型创建和参数统计
    import sys
    import os
    # 添加项目根目录到路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from utils import instantiate_from_config, print_model_parameters, load_config
    
    # 读取cifar10_unet.yaml配置文件
    config = load_config("../configs/cifar10_unet.yaml")
    
    # 使用utils函数实例化模型
    model = instantiate_from_config(config.trainer.params.model)
    
    # 打印模型参数量
    print_model_parameters(model)
    
    # 测试前向传播
    x = torch.randn(1, 3, 32, 32)
    t = torch.randn(1)
    
    with torch.no_grad():
        output = model(x, t)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
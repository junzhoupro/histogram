import os
from inspect import isfunction

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, c=None):
        h = self.heads

        q = self.to_q(x)
        c = default(c, x)
        k = self.to_k(c)
        v = self.to_v(c)
        # print(f"q: {q.shape}")
        # print(f"k: {k.shape}")
        # print(f"v: {v.shape}")

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == "fp32":
            with torch.autocast(enabled=False, device_type="cuda"):
                q, k = q.float(), k.float()
                sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
        else:
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        del q, k

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = torch.einsum("b i j, b j d -> b i d", sim, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, context_dim=None, dropout=0.0):
        super().__init__()
        self.attn = Attention(query_dim=dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, c=None):
        x = self.attn(self.norm1(x), c=c) + x
        x = self.ff(self.norm2(x)) + x
        return x


class Transformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head, mlp_ratio=4.0, depth=1, dropout=0.0, out_activation="relu", separated_channels=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        mlp_hidden_dim = int(inner_dim * mlp_ratio)
        self.proj_x = nn.Linear(in_channels, inner_dim)
        self.proj_c = nn.Linear(in_channels, inner_dim)
        self.norm_c = nn.LayerNorm(inner_dim)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(inner_dim, n_heads, d_head, mlp_hidden_dim, context_dim=inner_dim, dropout=dropout) for _ in range(depth)])
        self.proj_out = nn.Linear(inner_dim, in_channels)
        self.out_activation = out_activation
        assert out_activation in ["relu", "softmax", "log_softmax"]
        self.separated_channels = separated_channels

    def forward(self, x, c):
        x = x.permute(0, 2, 1)  # (N, C, B) => (N, B, C)
        c = c.permute(0, 2, 1)  # (N, C, B) => (N, B, C)

        if self.separated_channels:
            x_ = []
            for ch in range(x.shape[-1]):
                x_ch = self.proj_x(x[..., ch : ch + 1])
                c_ch = self.norm_c(self.proj_c(c[..., ch : ch + 1]))
                for block in self.transformer_blocks:
                    x_ch = block(x=x_ch, c=c_ch)
                x_.append(self.proj_out(x_ch))
            x = torch.concat(x_, dim=-1)
        else:
            x = self.proj_x(x)
            c = self.norm_c(self.proj_c(c))
            for block in self.transformer_blocks:
                x = block(x=x, c=c)
            x = self.proj_out(x)

        if self.out_activation == "relu":
            x = F.relu(x)
            x = x / (x.sum(dim=1, keepdim=True) + 1e-6)
        elif self.out_activation == "softmax":
            x = F.softmax(x, dim=1)
        else:
            x = F.log_softmax(x, dim=1)
            x = torch.exp(x)

        return x.permute(0, 2, 1)  # (N, B, C) => (N, C, B)

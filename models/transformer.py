import torch
import torch.nn as nn
import math
from einops import rearrange
from einops import repeat


class Residual(nn.Module):
    def __init__(self, fn, transformer_depth):
        super().__init__()
        self.fn = fn
        self.alpha = math.pow(2.0 * transformer_depth, 0.25)

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x * self.alpha

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = mask.flatten(1)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            mask = mask.bool()
            mask = repeat(mask, 'b n d -> b h n d', h=h)
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)
        attn = torch.nan_to_num(attn,0)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # self.layers.append(nn.ModuleList([
            #     Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout)), transformer_depth=depth),
            #     Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)), transformer_depth=depth)
            # ]))
            self.layers.append(nn.ModuleList([
                PostNorm(dim, Residual(Attention(dim, heads=heads, dropout=dropout), transformer_depth=depth)),
                PostNorm(dim, Residual(FeedForward(dim, mlp_dim, dropout=dropout), transformer_depth=depth))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

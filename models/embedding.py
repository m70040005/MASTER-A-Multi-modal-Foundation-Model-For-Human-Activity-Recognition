from torch import nn
from torch.autograd import Variable
import torch
import math
from einops import repeat

# position embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, configs, idx):
        super(PositionalEmbedding, self).__init__()
        dim = configs.embedding_dim
        max_len = configs.features_len[idx] * (configs.data_time_max_len[idx] // configs.chunk_length[idx]) + 1

        pe = torch.zeros(max_len, dim).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        b, n, _ = x.shape
        x = x + Variable(self.pe[:, 1:n+1,:], requires_grad=False)
        return x

# modal embedding
class ModuleEmbedding(nn.Module):
    def __init__(self, configs, idx):
        super(ModuleEmbedding, self).__init__()
        self.m_token = nn.Parameter(torch.randn(1, 1, 1))


    def forward(self, x):
        b, n, _ = x.shape
        m_tokens = repeat(self.m_token, '() () d -> b n d', b=b,n=n)
        x = torch.cat((m_tokens, x), dim=2)
        return x


class Embedding(nn.Module):
    '''
        In this part of model, we add position embeddings and modality embeddings to feature patches
    '''
    def __init__(self,configs):
        super(Embedding, self).__init__()
        self.modality_num = configs.modality_num
        self.modalities = configs.modalities
        self.embedding_dim = configs.embedding_dim
        self.empty_fill = configs.empty_fill
        self.p_emb = nn.ModuleList([PositionalEmbedding(configs,i) for i in range(self.modality_num)])
        self.m_emb = nn.ModuleList([ModuleEmbedding(configs,i) for i in range(self.modality_num)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim+1))

    def forward(self, features, input_mask):
        embedding_features = dict()
        for key in features.keys():
            idx = self.modalities.index(key)
            b, _, l = features[key].shape
            embedding_features[key] = self.p_emb[idx](features[key])
            embedding_features[key] = self.m_emb[idx](embedding_features[key])

            mask = input_mask[key].unsqueeze(2).repeat(1, 1, self.embedding_dim+1).bool()
            embedding_features[key] = embedding_features[key].masked_fill_(~mask,self.empty_fill)

        # add cls_token and its position embedding
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=b)
        pe = torch.zeros_like(cls_token)
        pe[:, :, 1::2] += 1
        cls_token = cls_token + Variable(pe, requires_grad=False)
        return embedding_features, cls_token

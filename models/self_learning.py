import torch
import torch.nn as nn
import numpy as np
from .transformer import Transformer

def generate_mask(inputs, length_mask, mask_fill ,mlm_probability=0.5):
    # TODO: Part of the reason why our code runs inefficiently is the "for" loop here,
    #      but we haven't found a suitable way to optimize it yet.

    b,c,z=inputs.shape
    # 50% of the data is masked and marked as 1, participating in the subsequent loss calculation,
    # while the remaining data is marked as 0
    mask = torch.ones([b,c])
    masked_indices = torch.bernoulli(torch.full(mask.shape, mlm_probability)).bool() & length_mask.bool().cpu()
    mask[~masked_indices] = 0
    # The same as Bert, for the masked data, 80% is assigned as a learnable block,
    # 10% is taken as a random other block, and the remaining 10% remains unchanged.
    indices_learned = torch.bernoulli(torch.full(mask.shape, 0.8)).bool()
    indices_random = torch.bernoulli(torch.full(mask.shape, 0.5)).bool() & ~indices_learned
    indices_learned = indices_learned & masked_indices
    indices_random = indices_random & masked_indices

    outputs = inputs.clone()
    for i in range(b):
        for j in range(c):
            if indices_learned[i, j]:
                outputs[i, j] = mask_fill
            elif indices_random[i, j]:
                limited_block = torch.nonzero(~masked_indices)
                rand_num = torch.randint(0, limited_block.shape[0],[1])
                outputs[i, j] = outputs[limited_block[rand_num][0][0], limited_block[rand_num][0][1]]

    return outputs, mask

class self_learning(nn.Module):
    '''
        In this part of model, we make embedded tokens into a sentence and put it into transformer.
        In self_learning mode, we use mask language modeling method to do self_learning train,
        so we mask some tokens before sending them into transformer.
    '''

    def __init__(self,configs,device):
        super(self_learning, self).__init__()
        self.device = device
        self.features_max_len = configs.features_max_len
        self.input_dim = configs.embedding_dim + 1
        self.mask_fill = nn.Parameter(torch.randn(self.input_dim))
        self.mlm_probability = configs.mlm_probability
        assert self.input_dim % configs.transformer_heads == 0, \
                "Config error: Embedding_dim+1 can't be divided by transformer_heads"
        self.transformer = Transformer(dim=self.input_dim, depth=configs.transformer_depth,
                                               heads=configs.transformer_heads, mlp_dim=self.input_dim * configs.mlp_ratio)

    def forward(self, features, length_mask, cls_token, training_mode="s"):
        # TODO: Part of the reason why our code runs inefficiently is the "for" loop below,
        #      but we haven't found a suitable way to optimize it yet.
        #      If you have a strict input length limitation within a batch, maybe you can avoid it.

        loss_mask = dict()
        masked_features = dict()
        if training_mode == "s":
            for key in features.keys():
                b, max_len, d = features[key].shape
                if length_mask[key].max() != 0:
                    masked_features[key], loss_mask[key] = generate_mask(features[key], length_mask[key],
                                                                         mask_fill=self.mask_fill,
                                                                         mlm_probability=self.mlm_probability)
                    loss_mask[key] = loss_mask[key].to(self.device)
                else:
                    loss_mask[key] = torch.zeros([b, max_len]).to(self.device)
                    masked_features[key] = features[key]
        elif training_mode == "a" or training_mode == 'f' or training_mode == 'am':
            for key in features.keys():
                b, max_len, d = features[key].shape
                loss_mask[key] = torch.zeros([b, max_len]).to(self.device)
                masked_features[key] = features[key]

        for bidx in range(0, b):
            forward_seq_now = cls_token[bidx, :, :]
            loss_mask_seq_now = torch.zeros_like(forward_seq_now)[:, 0]
            target_features_now = torch.zeros_like(forward_seq_now) + self.mask_fill
            length_mask_seq_now = torch.ones_like(forward_seq_now)[:, 0]
            for key in masked_features.keys():
                true_len = sum(length_mask[key][bidx, :])
                forward_seq_now = torch.cat((forward_seq_now, masked_features[key][bidx, :int(true_len), :]), dim=0)
                loss_mask_seq_now = torch.cat((loss_mask_seq_now, loss_mask[key][bidx, :int(true_len)]), dim=0)
                target_features_now = torch.cat((target_features_now, features[key][bidx, :int(true_len), :]),dim=0)
                length_mask_seq_now = torch.cat((length_mask_seq_now, length_mask[key][bidx, :int(true_len)]),dim=0)

            l = forward_seq_now.shape[0]
            assert l <= self.features_max_len, \
                "Config error: The forward sequence of transformer exceeds the set length in the config file"
            if l < self.features_max_len:
                forward_seq_now = torch.cat((forward_seq_now, torch.zeros(self.features_max_len - l, d).to(self.device) + self.mask_fill),dim=0)
                loss_mask_seq_now = torch.cat((loss_mask_seq_now, torch.zeros(self.features_max_len - l).to(self.device)), dim=0)
                target_features_now = torch.cat((target_features_now, torch.zeros(self.features_max_len - l, d).to(self.device) + self.mask_fill),dim=0)
                length_mask_seq_now = torch.cat((length_mask_seq_now, torch.zeros(self.features_max_len - l).to(self.device)), dim=0)

            if bidx == 0:
                forward_seq = forward_seq_now.unsqueeze(0)
                loss_mask_seq = loss_mask_seq_now.unsqueeze(0)
                target_features = target_features_now.unsqueeze(0)
                length_mask_seq = length_mask_seq_now.unsqueeze(0)
            else:
                forward_seq = torch.cat((forward_seq, forward_seq_now.unsqueeze(0)), dim=0)
                loss_mask_seq = torch.cat((loss_mask_seq, loss_mask_seq_now.unsqueeze(0)), dim=0)
                target_features = torch.cat((target_features, target_features_now.unsqueeze(0)), dim=0)
                length_mask_seq = torch.cat((length_mask_seq, length_mask_seq_now.unsqueeze(0)), dim=0)

        loss_mask_seq = loss_mask_seq.to(self.device)
        forward_seq = self.transformer(forward_seq, length_mask_seq)
        output_token = forward_seq[:, 0, :]

        return forward_seq[:, 1:, :], target_features[:, 1:, :], loss_mask_seq[:, 1:], output_token







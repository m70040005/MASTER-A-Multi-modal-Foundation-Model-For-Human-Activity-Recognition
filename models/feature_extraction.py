import torch
from torch import nn
from modality_process import *


class feature_extraction(nn.Module):
    '''
        In this part of model, we cut the input data of each modality into patches and
        send it to the corresponding feature extractor to extract feature patches
    '''
    def __init__(self, configs):
        super(feature_extraction, self).__init__()
        self.modalities = configs.modalities
        self.modality_num = configs.modality_num
        assert self.modality_num <= len(self.modalities), "Config error: modality_num or modalities in configs is wrong."

        self.feature_extraction_layer = nn.ModuleList()
        for i in range(0, self.modality_num):
            name = self.modalities[i]
            name_split = name.split('-')[0]
            if 'acc' in name_split or 'gyro' in name_split:
                self.feature_extraction_layer.append(imu_layers(configs, i))
            elif 'skeleton' in name_split:
                if '3d' in name_split:
                    self.feature_extraction_layer.append(skeleton_3d_layers(configs, i))
                elif '2d' in name_split:
                    self.feature_extraction_layer.append(skeleton_2d_layers(configs, i))
            elif 'mmwave' in name_split:
                if 'xyz' in name_split:
                    self.feature_extraction_layer.append(mmwave_xyz_layers(configs, i))
                elif 'di' in name_split:
                    self.feature_extraction_layer.append(mmwave_di_layers(configs, i))
            elif 'lidar' in name_split:
                self.feature_extraction_layer.append(lidar_layers(configs, i))
            elif 'wificsi' in name_split:
                self.feature_extraction_layer.append(wificsi_layers(configs, i))

        self.chunk_length = configs.chunk_length
        self.features_len = configs.features_len
        self.embedding_dim = configs.embedding_dim
        self.empty_fill = configs.empty_fill

    def forward(self, x_dict, input_mask):
        #TODO: Part of the reason why our code runs inefficiently is the "for" loop here,
        #      but we haven't found a suitable way to optimize it yet.
        #      If you use stricter standards during data preprocessing, you may be able to avoid it

        features = dict()
        output_mask = dict()
        for key in x_dict.keys():
            idx = self.modalities.index(key)
            key_split = key.split('-')[0]
            if 'acc' in key_split or 'gyro' in key_split:
                b, c, t = x_dict[key].shape
                device = x_dict[key].device
                features[key] = torch.zeros([b, self.embedding_dim, self.features_len[idx] * (t // self.chunk_length[idx])]).to(device)
                output_mask[key] = torch.ones([b, self.features_len[idx] * (t // self.chunk_length[idx])]).to(device)
                for chunk_id in range(0, t // self.chunk_length[idx]):
                    features[key][:, :, chunk_id * self.features_len[idx]:chunk_id * self.features_len[idx] + self.features_len[idx]] = \
                        self.feature_extraction_layer[idx](x_dict[key][:, :,chunk_id * self.chunk_length[idx]:chunk_id * self.chunk_length[idx] +self.chunk_length[idx]])
                    for bidx in range(0, b):
                        if sum(input_mask[key][bidx,chunk_id * self.chunk_length[idx]:chunk_id * self.chunk_length[idx] + self.chunk_length[idx]]) == 0:
                            output_mask[key][bidx, chunk_id * self.features_len[idx]:chunk_id * self.features_len[idx] + self.features_len[idx]] = 0
                mask = output_mask[key].unsqueeze(2).repeat(1, 1, self.embedding_dim).bool()
                features[key] = features[key].transpose(1, 2).masked_fill_(~mask, self.empty_fill)
            elif 'skeleton' in key_split:
                b, t, n, c = x_dict[key].shape
                device = x_dict[key].device
                features[key] = torch.zeros([b, self.embedding_dim, self.features_len[idx] * (t // self.chunk_length[idx])]).to(device)
                output_mask[key] = torch.ones([b, self.features_len[idx] * (t // self.chunk_length[idx])]).to(device)
                for chunk_id in range(0, t // self.chunk_length[idx]):
                    features[key][:, :, chunk_id * self.features_len[idx]:chunk_id * self.features_len[idx] + self.features_len[idx]] = \
                        self.feature_extraction_layer[idx](x_dict[key][:, chunk_id * self.chunk_length[idx]:chunk_id * self.chunk_length[idx] + self.chunk_length[idx], :, :])
                    for bidx in range(0, b):
                        if sum(input_mask[key][bidx,chunk_id * self.chunk_length[idx]:chunk_id * self.chunk_length[idx] + self.chunk_length[idx]]) == 0:
                            output_mask[key][bidx, chunk_id * self.features_len[idx]:chunk_id * self.features_len[idx] + self.features_len[idx]] = 0
                mask = output_mask[key].unsqueeze(2).repeat(1, 1, self.embedding_dim).bool()
                features[key] = features[key].transpose(1, 2).masked_fill_(~mask, self.empty_fill)
            elif 'mmwave' in key_split:
                if 'xyz' in key:
                    b, t, c, w, h = x_dict[key].shape
                    device = x_dict[key].device
                    features[key] = torch.zeros(
                        [b, self.embedding_dim, self.features_len[idx] * (t // self.chunk_length[idx])]).to(device)
                    output_mask[key] = torch.ones([b, self.features_len[idx] * (t // self.chunk_length[idx])]).to(device)
                    for chunk_id in range(0, t // self.chunk_length[idx]):
                        features[key][:, :,
                        chunk_id * self.features_len[idx]:chunk_id * self.features_len[idx] + self.features_len[idx]] = \
                            self.feature_extraction_layer[idx](x_dict[key][:,chunk_id * self.chunk_length[idx]:chunk_id * self.chunk_length[idx] +self.chunk_length[idx], :, :, :])
                        for bidx in range(0, b):
                            if sum(input_mask[key][bidx,chunk_id * self.chunk_length[idx]:chunk_id * self.chunk_length[idx] +self.chunk_length[idx]]) == 0:
                                output_mask[key][bidx,chunk_id * self.features_len[idx]:chunk_id * self.features_len[idx] + self.features_len[idx]] = 0
                    mask = output_mask[key].unsqueeze(2).repeat(1, 1, self.embedding_dim).bool()
                    features[key] = features[key].transpose(1, 2).masked_fill_(~mask, self.empty_fill)
                elif 'di' in key_split:
                    b, t, c = x_dict[key].shape
                    device = x_dict[key].device
                    features[key] = torch.zeros([b, self.embedding_dim, self.features_len[idx] * (t // self.chunk_length[idx])]).to(device)
                    output_mask[key] = torch.ones([b, self.features_len[idx] * (t // self.chunk_length[idx])]).to(device)
                    for chunk_id in range(0, t // self.chunk_length[idx]):
                        features[key][:, :,chunk_id * self.features_len[idx]:chunk_id * self.features_len[idx] + self.features_len[idx]] = \
                            self.feature_extraction_layer[idx](x_dict[key][:,chunk_id * self.chunk_length[idx]:chunk_id * self.chunk_length[idx] +self.chunk_length[idx], :])
                        for bidx in range(0, b):
                            if sum(input_mask[key][bidx,chunk_id * self.chunk_length[idx]:chunk_id * self.chunk_length[idx] +self.chunk_length[idx]]) == 0:
                                output_mask[key][bidx,chunk_id * self.features_len[idx]:chunk_id * self.features_len[idx] + self.features_len[idx]] = 0
                    mask = output_mask[key].unsqueeze(2).repeat(1, 1, self.embedding_dim).bool()
                    features[key] = features[key].transpose(1, 2).masked_fill_(~mask, self.empty_fill)
            elif 'lidar' in key_split:
                b, t, c, w, h = x_dict[key].shape
                device = x_dict[key].device
                features[key] = torch.zeros(
                    [b, self.embedding_dim, self.features_len[idx] * (t // self.chunk_length[idx])]).to(device)
                output_mask[key] = torch.ones([b, self.features_len[idx] * (t // self.chunk_length[idx])]).to(device)
                for chunk_id in range(0, t // self.chunk_length[idx]):
                    features[key][:, :,chunk_id * self.features_len[idx]:chunk_id * self.features_len[idx] + self.features_len[idx]] = \
                        self.feature_extraction_layer[idx](x_dict[key][:,chunk_id * self.chunk_length[idx]:chunk_id * self.chunk_length[idx] + self.chunk_length[idx], :, :, :])
                    for bidx in range(0, b):
                        if sum(input_mask[key][bidx,chunk_id * self.chunk_length[idx]:chunk_id * self.chunk_length[idx] + self.chunk_length[idx]]) == 0:
                            output_mask[key][bidx,chunk_id * self.features_len[idx]:chunk_id * self.features_len[idx] + self.features_len[idx]] = 0
                mask = output_mask[key].unsqueeze(2).repeat(1, 1, self.embedding_dim).bool()
                features[key] = features[key].transpose(1, 2).masked_fill_(~mask, self.empty_fill)
            elif 'wificsi' in key_split:
                b, t, c, n = x_dict[key].shape
                device = x_dict[key].device
                features[key] = torch.zeros([b, self.embedding_dim, self.features_len[idx] * (t // self.chunk_length[idx])]).to(device)
                output_mask[key] = torch.ones([b, self.features_len[idx] * (t // self.chunk_length[idx])]).to(device)
                for chunk_id in range(0, t // self.chunk_length[idx]):
                    features[key][:, :,chunk_id * self.features_len[idx]:chunk_id * self.features_len[idx] + self.features_len[idx]] = \
                        self.feature_extraction_layer[idx](x_dict[key][:, chunk_id * self.chunk_length[idx]:chunk_id * self.chunk_length[idx] +self.chunk_length[idx], :, :])
                    for bidx in range(0, b):
                        if sum(input_mask[key][bidx,chunk_id * self.chunk_length[idx]:chunk_id * self.chunk_length[idx] + self.chunk_length[idx]]) == 0:
                            output_mask[key][bidx,chunk_id * self.features_len[idx]:chunk_id * self.features_len[idx] + self.features_len[idx]] = 0
                mask = output_mask[key].unsqueeze(2).repeat(1, 1, self.embedding_dim).bool()
                features[key] = features[key].transpose(1, 2).masked_fill_(~mask, self.empty_fill)
        return features, output_mask

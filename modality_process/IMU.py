import numpy as np
import torch
from torch import nn


def imu_preprocess(dataset_dict, modality, data_num, max_len, empty_fill):
    assert data_num > 0
    if modality in dataset_dict:
        x_data = dataset_dict[modality]
        if data_num == 1:
            t_len = x_data.shape[1]
            if t_len < max_len:
                tmp = torch.zeros([3, max_len - t_len])
                input_mask = torch.cat([torch.ones([t_len]),
                                        torch.zeros([max_len - t_len])], dim=0)
                x_data = torch.cat([x_data, tmp + empty_fill], dim=1)
            elif t_len >= max_len:
                input_mask = torch.ones([max_len])
                x_data = x_data[:, :max_len]
        elif type(x_data) == torch.Tensor:
            t_len = x_data.shape[2]
            if t_len < max_len:
                tmp = torch.zeros([data_num, 3, max_len - t_len])
                input_mask = torch.cat([torch.ones([data_num, t_len]),
                                        torch.zeros([data_num, max_len - t_len])], dim=1)
                x_data = torch.cat([x_data, tmp + empty_fill], dim=2)
            elif t_len >= max_len:
                input_mask = torch.ones([data_num, max_len])
                x_data = x_data[:, :, :max_len]
        elif type(x_data) == list:
            input_mask = []
            for idx in range(0, data_num):
                t_len = x_data[idx].shape[1]
                if t_len < max_len:
                    tmp = np.zeros([3, max_len - t_len])
                    input_mask.append(np.concatenate([np.ones(t_len), np.zeros(max_len - t_len)], axis=0))
                    x_data[idx] = np.concatenate([x_data[idx], tmp + empty_fill], axis=1)
                elif t_len >= max_len:
                    input_mask.append(np.ones(max_len))
                    x_data[idx] = x_data[idx][:, :max_len]
            x_data = torch.tensor(np.array(x_data))
            input_mask = torch.tensor(np.array(input_mask))
    else:
        if data_num == 1:
            input_mask = torch.zeros([max_len])
            x_data = torch.zeros([3, max_len]) + empty_fill
        else:
            input_mask = torch.zeros([data_num, max_len])
            x_data = torch.zeros([data_num, 3, max_len]) + empty_fill
    if type(x_data) != torch.Tensor:
        x_data = torch.tensor(x_data)
    return x_data, input_mask

class imu_layers(nn.Module):
    def __init__(self, configs, idx):
        super(imu_layers, self).__init__()
        self.dim = configs.embedding_dim
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=8,
                      stride=1, bias=False, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.layer = nn.Linear(640, configs.embedding_dim * configs.features_len[idx])

    def forward(self, x_in):
        b = x_in.shape[0]
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = self.layer(x.reshape(b,-1))
        x = x.reshape(b, self.dim, -1)
        return x
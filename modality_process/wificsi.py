from torch import nn
import numpy as np
import torch

def wificsi_preprocess(dataset_dict, modality, data_num, max_len, empty_fill):
    assert data_num > 0
    if modality in dataset_dict:
        x_data = dataset_dict[modality]
        if data_num == 1:
            t_len, c, wifisenser_num = x_data.shape
            if t_len < max_len:
                tmp = torch.zeros([max_len - t_len, c, wifisenser_num])
                input_mask = torch.cat([torch.ones([t_len]),
                                        torch.zeros([max_len - t_len])], dim=0)
                x_data = torch.cat([x_data, tmp + empty_fill], dim=0)
            elif t_len >= max_len:
                input_mask = torch.ones([max_len])
                x_data = x_data[:max_len, :, :]
        elif type(x_data) == torch.Tensor:
            _, t_len, c, wifisenser_num = x_data.shape
            if t_len < max_len:
                tmp = torch.zeros([data_num, max_len - t_len, c, wifisenser_num])
                input_mask = torch.cat([torch.ones([data_num, t_len]),
                                        torch.zeros([data_num, max_len - t_len])], dim=1)
                x_data = torch.cat([x_data, tmp + empty_fill], dim=1)
            elif t_len >= max_len:
                input_mask = torch.ones([data_num, max_len])
                x_data = x_data[:, :max_len, :, :]
        elif type(x_data) == list:
            input_mask = []
            _, c, wifisenser_num = x_data[0].shape
            for idx in range(0, data_num):
                t_len = x_data[idx].shape[0]
                if t_len < max_len:
                    tmp = np.zeros([max_len - t_len, c, wifisenser_num])
                    input_mask.append(np.concatenate([np.ones(t_len), np.zeros(max_len - t_len)], axis=0))
                    x_data[idx] = np.concatenate([x_data[idx], tmp + empty_fill], axis=0)
                elif t_len >= max_len:
                    input_mask.append(np.ones(max_len))
                    x_data[idx] = x_data[idx][:max_len, :, :]
            x_data = torch.tensor(np.array(x_data))
            input_mask = torch.tensor(np.array(input_mask))
    else:
        if data_num == 1:
            input_mask = torch.zeros([max_len])
            x_data = torch.zeros([max_len, 3, 114]) + empty_fill
        else:
            input_mask = torch.zeros([data_num, max_len])
            x_data = torch.zeros([data_num, max_len, 3, 114]) + empty_fill

    return x_data, input_mask

class wificsi_layers(nn.Module):

    def __init__(self, configs, idx):
        super(wificsi_layers, self).__init__()
        self.dim = configs.embedding_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 12, 6),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 7, 3),
            nn.ReLU(),
        )
        self.mean = nn.AvgPool1d(32)
        self.gru = nn.GRU(8, 128, num_layers=1)
        self.layer = nn.Linear(128, configs.features_len[idx] * configs.embedding_dim)


    def forward(self, x_in):
        batch_size, t_len, c, sn = x_in.shape
        # batch x t_len x 3 x 114
        x = x_in.reshape(batch_size * t_len, c * sn).unsqueeze(1)
        # (batch x t_len) x 1 x (3 x 114)
        x = self.encoder(x)
        # (batch x t_len) x 32 x 6
        x = x.transpose(2, 1)
        x = self.mean(x)
        x = x.reshape(batch_size, t_len, 8)
        # batch x t_len x 6
        x = x.transpose(1, 0)
        # t_len x batch x 6
        _, x = self.gru(x)
        x = x.transpose(1, 0)
        # batch x 1 x 128
        x = self.layer(x.reshape(batch_size, -1))
        x = x.reshape(batch_size,self.dim,-1)
        # batch x dim x feature_len
        return x


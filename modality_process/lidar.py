from torch import nn
import torch
import numpy as np

def lidar_preprocess(dataset_dict, modality, data_num, max_len, empty_fill):
    assert data_num > 0
    if modality in dataset_dict:
        x_data = dataset_dict[modality]
        if data_num == 1:
            t_len, c, w, h = x_data.shape
            if t_len < max_len:
                tmp = torch.zeros([max_len - t_len, c, w, h])
                input_mask = torch.cat([torch.ones([t_len]),
                                        torch.zeros([max_len - t_len])], dim=0)
                x_data = torch.cat([x_data, tmp + empty_fill], dim=0)
            elif t_len >= max_len:
                input_mask = torch.ones([max_len])
                x_data = x_data[:max_len, :, :, :]
        elif type(x_data) == torch.Tensor:
            _, t_len, c, w, h = x_data.shape
            if t_len < max_len:
                tmp = torch.zeros([data_num, max_len - t_len, c, w, h])
                input_mask = torch.cat([torch.ones([data_num, t_len]),
                                        torch.zeros([data_num, max_len - t_len])], dim=1)
                x_data = torch.cat([x_data, tmp + empty_fill], dim=1)
            elif t_len >= max_len:
                input_mask = torch.ones([data_num, max_len])
                x_data = x_data[:, :max_len, :, :, :]
        elif type(x_data) == list:
            input_mask = []
            _, c, w, h = x_data[0].shape
            for idx in range(0, data_num):
                t_len = x_data[idx].shape[0]
                if t_len < max_len:
                    tmp = np.zeros([max_len - t_len, c, w, h])
                    input_mask.append(np.concatenate([np.ones(t_len), np.zeros(max_len - t_len)], axis=0))
                    x_data[idx] = np.concatenate([x_data[idx], tmp + empty_fill], axis=0)
                elif t_len >= max_len:
                    input_mask.append(np.ones(max_len))
                    x_data[idx] = x_data[idx][:max_len, :, :, :]
            x_data = torch.tensor(np.array(x_data))
            input_mask = torch.tensor(np.array(input_mask))
    else:
        if data_num == 1:
            input_mask = torch.zeros([max_len])
            x_data = torch.zeros([max_len, 3, 60, 60]) + empty_fill
        else:
            input_mask = torch.zeros([data_num, max_len])
            x_data = torch.zeros([data_num, max_len, 3, 60, 60]) + empty_fill

    return x_data, input_mask

class lidar_layers(nn.Module):
    def __init__(self, configs, idx):
        super(lidar_layers, self).__init__()
        self.dim = configs.embedding_dim
        self.relu = nn.ReLU(inplace=True)
        self.block1 = nn.Sequential(
            nn.Conv3d(configs.input_channels[idx], 64, kernel_size=(3, 7, 7),
                      stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64),
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64),
        )
        self.block4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64),
        )
        self.block5 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64),
        )
        self.block6 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=(3, 5, 5), stride=2, padding=(0, 2, 2), bias=False),
            nn.BatchNorm3d(32),
        )
        self.layer = nn.Linear(4096, configs.features_len[idx] * configs.final_out_channels)
    def forward(self, x_in):
        # batch x t_len x c x w x h
        b = x_in.shape[0]
        x = x_in.transpose(1,2)
        # batch x c x t_len x w x h
        x = self.block1(x)

        r = x
        x = self.block2(x)
        x = self.relu(x)
        x = self.block3(x)
        x = self.relu(x + r)

        r = x
        x = self.block4(x)
        x = self.relu(x)
        x = self.block5(x)
        x = self.relu(x + r)

        x = self.block6(x)
        x = self.layer(x.reshape(b, -1))
        x = x.reshape(b, self.dim, -1)

        return x
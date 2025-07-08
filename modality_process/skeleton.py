import numpy as np
import torch
from torch import nn

def skeleton_preprocess(dataset_dict, modality, data_num, max_len, empty_fill):
    assert data_num > 0
    if modality in dataset_dict:
        x_data = dataset_dict[modality]
        if data_num == 1:
            t_len, point_num, dim_num = x_data.shape
            if t_len < max_len:
                tmp = torch.zeros([max_len - t_len, point_num, dim_num])
                input_mask = torch.cat([torch.ones([t_len]),
                                        torch.zeros([max_len - t_len])], dim=0)
                x_data = torch.cat([x_data, tmp + empty_fill], dim=0)
            elif t_len >= max_len:
                input_mask = torch.ones([max_len])
                x_data = x_data[:max_len, :, :]
        elif type(x_data) == torch.Tensor:
            _, t_len, point_num, dim_num = x_data.shape
            if t_len < max_len:
                tmp = torch.zeros([data_num, max_len - t_len, point_num, dim_num])
                input_mask = torch.cat([torch.ones([data_num, t_len]),
                                        torch.zeros([data_num, max_len - t_len])], dim=1)
                x_data = torch.cat([x_data, tmp + empty_fill], dim=1)
            elif t_len >= max_len:
                input_mask = torch.ones([data_num, max_len])
                x_data = x_data[:, :max_len, :, :]
        elif type(x_data) == list:
            input_mask = []
            _, point_num, dim_num = x_data[0].shape
            for idx in range(0, data_num):
                t_len = x_data[idx].shape[0]
                if t_len < max_len:
                    tmp = np.zeros([max_len - t_len, point_num, dim_num])
                    input_mask.append(np.concatenate([np.ones(t_len), np.zeros(max_len - t_len)], axis=0))
                    x_data[idx] = np.concatenate([x_data[idx], tmp + empty_fill], axis=0)
                elif t_len >= max_len:
                    input_mask.append(np.ones(max_len))
                    x_data[idx] = np.array(x_data[idx][:max_len, :, :])
            x_data = torch.tensor(np.array(x_data))
            input_mask = torch.tensor(np.array(input_mask))
    else:
        if data_num == 1:
            input_mask = torch.zeros([max_len])
            if "2d" in modality:
                x_data = torch.zeros([max_len, 17, 2]) + empty_fill
            elif "3d" in modality:
                x_data = torch.zeros([max_len, 17, 3]) + empty_fill
        else:
            input_mask = torch.zeros([data_num, max_len])
            if "2d" in modality:
                x_data = torch.zeros([data_num, max_len, 17, 2]) + empty_fill
            elif "3d" in modality:
                x_data = torch.zeros([data_num, max_len, 17, 3]) + empty_fill
    return x_data, input_mask

class skeleton_3d_layers(nn.Module):
    def __init__(self, configs, idx):
        super(skeleton_3d_layers, self).__init__()
        self.dim = configs.embedding_dim
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5,5,2), padding=(2,0,0), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(5,5,2), padding=(2,0,0), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=(5,5,1), padding=(1,0,0), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=(5,2,1), padding=(1,0,0), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.layer = nn.Linear(192, configs.embedding_dim * configs.features_len[idx])

    def forward(self, x_in):  # x[batch_size 40 20 3]
        batch_size = x_in.shape[0]
        x = x_in.unsqueeze(1) # x[batch_size 1 40 20 3]

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = self.layer(x.reshape(batch_size,-1))
        x = x.reshape(batch_size,self.dim,-1)
        return x

class skeleton_proj(nn.Module):
    def __init__(self, configs, idx):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(8, 64, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, configs.embedding_dim, 1),
            nn.BatchNorm2d(configs.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        b,t,n,c = x.shape
        x = x.reshape(b*5,int(t/5),n,c)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(b,5,-1).transpose(1,2)
        return x

class skeleton_2d_layers(nn.Module):
    def __init__(self, configs, idx):
        super(skeleton_2d_layers, self).__init__()
        self.dim = configs.embedding_dim
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3,5,2), padding=(1,0,0), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3,5,1), padding=(1,0,0), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=(3,5,1), padding=(1,0,0), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=(3,2,1), padding=(1,0,0), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.layer = nn.Linear(256, configs.embedding_dim * configs.features_len[idx])

    def forward(self, x_in):  # x[batch_size 40 20 2]
        batch_size = x_in.shape[0]
        x = x_in.unsqueeze(1) # x[batch_size 1 40 20 2]

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = self.layer(x.reshape(batch_size,-1))
        x = x.reshape(batch_size,self.dim,-1)
        return x
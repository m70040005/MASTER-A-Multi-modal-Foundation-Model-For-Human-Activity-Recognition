from torch import nn
import torch
import numpy as np

def mmwave_preprocess(dataset_dict, modality, data_num, max_len, empty_fill):
    assert data_num > 0
    if modality in dataset_dict:
        x_data = dataset_dict[modality]
        if data_num == 1:
            if 'xyz' in modality:
                t_len, c, w, h = x_data.shape
                if t_len < max_len:
                    tmp = torch.zeros([max_len - t_len, c, w, h])
                    input_mask = torch.cat([torch.ones([t_len]),
                                            torch.zeros([max_len - t_len])], dim=0)
                    x_data = torch.cat([x_data, tmp + empty_fill],dim=0)
                elif t_len >= max_len:
                    input_mask = torch.ones([max_len])
                    x_data = x_data[:max_len, :, :, :]
            elif 'di' in modality:
                t_len, c = x_data.shape
                if t_len < max_len:
                    tmp = torch.zeros([max_len - t_len, c])
                    input_mask = torch.cat([torch.ones([t_len]),
                                            torch.zeros([max_len - t_len])], dim=0)
                    x_data = torch.cat([x_data, tmp + empty_fill], dim=0)
                elif t_len >= max_len:
                    input_mask = torch.ones([max_len])
                    x_data = x_data[:max_len, :]
        elif type(x_data) == torch.Tensor:
            if 'xyz' in modality:
                _, t_len, c, w, h = x_data.shape
                if t_len < max_len:
                    tmp = torch.zeros([data_num, max_len - t_len, c, w, h])
                    input_mask = torch.cat([torch.ones([data_num, t_len]),
                                            torch.zeros([data_num, max_len - t_len])], dim=1)
                    x_data = torch.cat([x_data, tmp + empty_fill],dim=1)
                elif t_len >= max_len:
                    input_mask = torch.ones([data_num, max_len])
                    x_data = x_data[:, :max_len, :, :, :]
            elif 'di' in modality:
                _, t_len, c = x_data.shape
                if t_len < max_len:
                    tmp = torch.zeros([data_num, max_len - t_len, c])
                    input_mask = torch.cat([torch.ones([data_num, t_len]),
                                            torch.zeros([data_num, max_len - t_len])], dim=1)
                    x_data = torch.cat([x_data, tmp + empty_fill], dim=1)
                elif t_len >= max_len:
                    input_mask = torch.ones([data_num, max_len])
                    x_data = x_data[:, :max_len, :]
        elif type(x_data) == list:
            input_mask = []
            if 'xyz' in modality:
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
            elif 'di' in modality:
                _, c = x_data[0].shape
                for idx in range(0, data_num):
                    t_len = x_data[idx].shape[0]
                    if t_len < max_len:
                        tmp = np.zeros([data_num, max_len - t_len, c])
                        input_mask.append(np.concatenate([np.ones(t_len), np.zeros(max_len - t_len)], axis=0))
                        x_data[idx] = np.concatenate([x_data[idx], tmp + empty_fill], axis=0)
                    elif t_len >= max_len:
                        input_mask.append(np.ones(max_len))
                        x_data[idx] = x_data[idx][:max_len, :]
            x_data = torch.tensor(np.array(x_data))
            input_mask = torch.tensor(np.array(input_mask))
    else:
        if data_num == 1:
            if 'xyz' in modality:
                input_mask = torch.zeros([max_len])
                x_data = torch.zeros([max_len, 3, 60, 60]) + empty_fill
            elif 'di' in modality:
                input_mask = torch.zeros([max_len])
                x_data = torch.zeros([max_len, 17]) + empty_fill
        else:
            if 'xyz' in modality:
                input_mask = torch.zeros([data_num, max_len])
                x_data = torch.zeros([data_num, max_len, 3, 60, 60]) + empty_fill
            elif 'di' in modality:
                input_mask = torch.zeros([data_num, max_len])
                x_data = torch.zeros([data_num, max_len, 17]) + empty_fill
    return x_data, input_mask

class mmwave_di_layers(nn.Module):
    def __init__(self, configs, idx):
        super(mmwave_di_layers, self).__init__()
        self.dim = configs.embedding_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.fc4 = nn.Linear(192, configs.features_len[idx] * configs.embedding_dim)

    def forward(self, x_in):
        # batch x t_len x 17
        b = x_in.shape[0]
        x = x_in.unsqueeze(1)
        # batch x 1 x t_len x 17
        x = self.conv1(x)
        x = self.relu(x)
        # batch x 64 x t_len1 x 9
        x = self.conv2(x)
        x = self.maxpool(x)
        # batch x 64 x t_len2 x 3
        x = self.conv3(x)
        x = self.bn1(x)
        # batch x 64 x t_len2 x 3
        x = self.fc4(x.reshape(b,-1))
        x = x.reshape(b, self.dim, -1)
        # batch x dim x feature_len
        return x


# class mmwave_di_layers2(nn.Module):
#     def __init__(self, configs, idx):
#         super().__init__()
#         self.dim = configs.embedding_dim
#
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 64, 3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(64, 128, 3),
#             nn.Dropout(0.1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(128, 256, 1, stride=1),
#             nn.Dropout(0.2),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(256, configs.features_len[idx] * configs.embedding_dim, 1),
#             nn.Dropout(0.3),
#             nn.BatchNorm2d(configs.features_len[idx] * configs.embedding_dim),
#             nn.ReLU(inplace=True),
#         )
#
#         self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
#
#     def forward(self, x):
#         b = x.shape[0]
#         x1 = x.unsqueeze(1)
#         x1 = self.features(x1)
#         x1 = self.avgpool(x1)
#         x1 = torch.flatten(x1, 1)
#         x1 = x1.reshape(b, self.dim, -1)
#
#         return x1

class mmwave_xyz_layers(nn.Module):
    def __init__(self, configs, idx):
        super(mmwave_xyz_layers, self).__init__()
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
        self.layer = nn.Linear(4096, configs.features_len[idx] * configs.embedding_dim)
    def forward(self, x_in):
        # batch x t_len x 3 x w x h
        b = x_in.shape[0]
        x = x_in.transpose(1,2)
        # batch x 3 x t_len x w x h
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
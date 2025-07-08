import os
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from modality_process import *

class Load_Dataset(Dataset):

    def __init__(self, dataset, configs):
        super(Load_Dataset, self).__init__()
        self.modality_list = configs.modalities[:configs.modality_num]
        self.max_len = configs.data_time_max_len
        self.empty_fill = configs.empty_fill
        self.x_data = dict()
        self.input_mask = dict()

        # Integrating data from multiple datasets
        for dataset_idx in range(0, len(dataset)):
            x_data = dict()
            input_mask = dict()
            data_num = dataset[dataset_idx]['labels'].shape[0]

            modality_idx = 0
            for modality in self.modality_list:
                modality_split = modality.split('-')[0]
                if 'acc' in modality_split or 'gyro' in modality_split:
                    x_data[modality], input_mask[modality] = imu_preprocess(dataset[dataset_idx], modality,
                                                                            data_num, self.max_len[modality_idx],
                                                                            self.empty_fill)
                elif 'skeleton' in modality_split:
                    x_data[modality], input_mask[modality] = skeleton_preprocess(dataset[dataset_idx], modality,
                                                                                 data_num, self.max_len[modality_idx],
                                                                                 self.empty_fill)
                elif 'mmwave' in modality_split:
                    x_data[modality], input_mask[modality] = mmwave_preprocess(dataset[dataset_idx], modality,
                                                                               data_num, self.max_len[modality_idx],
                                                                               self.empty_fill)
                elif 'lidar' in modality_split:
                    x_data[modality], input_mask[modality] = lidar_preprocess(dataset[dataset_idx], modality,
                                                                              data_num, self.max_len[modality_idx],
                                                                              self.empty_fill)

                elif 'wificsi' in modality_split:
                    x_data[modality], input_mask[modality] = wificsi_preprocess(dataset[dataset_idx], modality,
                                                                                data_num, self.max_len[modality_idx],
                                                                                self.empty_fill)
                modality_idx += 1
            y_data = torch.tensor(dataset[dataset_idx]["labels"])

            if dataset_idx == 0:
                for modality in self.modality_list:
                    self.x_data[modality] = x_data[modality]
                    self.input_mask[modality] = input_mask[modality]
                self.dfrom = torch.zeros_like(y_data)
                self.y_data = y_data - y_data.min()
            else:
                for modality in self.modality_list:
                    self.x_data[modality] = torch.cat([self.x_data[modality], x_data[modality]], dim=0)
                    self.input_mask[modality] = torch.cat([self.input_mask[modality], input_mask[modality]],
                                                          dim=0)
                self.dfrom = torch.cat([self.dfrom, torch.zeros_like(y_data) + dataset_idx], dim=0)
                self.y_data = torch.cat([self.y_data, y_data - y_data.min() + self.y_data.max() + 1], dim=0)

        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        y_data = self.y_data[index]
        dfrom = self.dfrom[index]
        x_data = dict()
        input_mask = dict()
        for key in self.x_data.keys():
            x_data[key] = self.x_data[key][index]
            input_mask[key] = self.input_mask[key][index]
        return x_data, y_data, input_mask, dfrom

    def __len__(self):
        return self.len

def data_generator(data_path, configs, logger, training_mode, label_rate):
    train_dataset = []
    test_dataset = []

    if (training_mode == 's' or training_mode == 'am') and label_rate != 1:
        train_data_path = data_path + '/train_' + str(label_rate)
        if os.path.exists(os.path.join(train_data_path, "train_unlabel.pt")):
            train_dataset.append(torch.load(os.path.join(train_data_path, "train_unlabel.pt")))
    elif (training_mode == 'a' or training_mode == 'f') and label_rate != 1:
        train_data_path = data_path + '/train_' + str(label_rate)
        if os.path.exists(os.path.join(train_data_path, "train_label.pt")):
            train_dataset.append(torch.load(os.path.join(train_data_path, "train_label.pt")))
    else:
        if os.path.exists(os.path.join(data_path, "train.pt")):
            train_dataset.append(torch.load(os.path.join(data_path, "train.pt")))

    if os.path.exists(os.path.join(data_path, "test.pt")):
        test_dataset.append(torch.load(os.path.join(data_path, "test.pt")))

    train_dataset = Load_Dataset(train_dataset, configs)
    test_dataset = Load_Dataset(test_dataset, configs)
    logger.debug(f'train dataset length: {train_dataset.len}\n'
                 f'test dataset length: {test_dataset.len}')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=False,
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, test_loader

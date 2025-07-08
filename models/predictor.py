import torch.nn as nn
import torch

class predictor(nn.Module):
    '''
        In this part of model, we use cls_token to predict class label
    '''
    def __init__(self, configs):
        super(predictor, self).__init__()
        self.num_classes = configs.num_classes
        self.embedding_dim = configs.embedding_dim + 1
        self.align_dim = self.embedding_dim * configs.align_ratio

        self.feature_proj_block = nn.Sequential(
            nn.Linear(self.embedding_dim, self.align_dim),
            nn.Linear(self.align_dim, self.embedding_dim),
        )
        self.drop = nn.Dropout(configs.pred_dropout)
        self.predictlayer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.num_classes),
        )


    def forward(self, input):
        output = self.feature_proj_block(input.flatten(1))
        output = self.drop(output)
        output = self.predictlayer(output)

        return output
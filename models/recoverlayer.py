import torch.nn as nn
class recoverlayer(nn.Module):
    '''
        In this part of model, we recover masked tokens from transformer output tokens
    '''
    def __init__(self,configs):
        super(recoverlayer, self).__init__()
        self.embedding_dim = configs.embedding_dim + 1
        self.hidden_dim = self.embedding_dim

        self.recoverlayer = nn.Linear(self.hidden_dim, self.embedding_dim)

    def forward(self,forward_seq):

        predict_features = self.recoverlayer(forward_seq)

        return predict_features
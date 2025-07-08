import torch
import torch.nn as nn

# Our self_learning loss. In fact, we found it is a different implement of InfoNCEloss.
class selfLearningLoss(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(selfLearningLoss, self).__init__()
        self.temperature = temperature
        self.lsoftmax = nn.LogSoftmax(dim=0)

    def forward(self, transformer_outputs, target_features, loss_mask):

        similarity_matrix = torch.matmul(
            torch.flatten(transformer_outputs, start_dim=0, end_dim=1),
            torch.transpose(torch.flatten(target_features, start_dim=0, end_dim=1), 0, 1)
        )
        similarity_matrix /= self.temperature
        similarity_matrix = torch.diag(self.lsoftmax(similarity_matrix))

        loss_mask = (loss_mask == 1).reshape(-1)
        loss = -1. * sum(similarity_matrix[loss_mask]) / sum(loss_mask)

        return loss

# Alignment loss:
# We use it to align all cls_token from different scenes, positions, and devices
class alignLearningLoss(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(alignLearningLoss, self).__init__()
        self.temperature = temperature
        self.lsoftmax = nn.LogSoftmax(dim=0)

    def forward(self, tokens, labels):
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(1)
        _, modality_num, _ = tokens.shape
        loss = 0
        num = 0
        for i in range(0, labels.shape[0]):
            same_label = (labels == labels[i])
            positive = torch.nonzero(same_label).squeeze()
            negative = torch.nonzero(~same_label).squeeze()
            for j in range(0,modality_num):
                similarity_matrix = torch.matmul(tokens[i,j].unsqueeze(0),
                                                 torch.cat([tokens[positive[positive != i],j], tokens[negative,j]], dim=0).T
                                                 ).squeeze()
                similarity_matrix /= self.temperature
                similarity_matrix = self.lsoftmax(similarity_matrix)
                if sum(same_label) > 1:
                    positive_num = positive.shape[0] - 1
                    loss += sum(similarity_matrix[:positive_num]) / positive_num
                    num += 1

        return loss/(-1. * num)

# Modality missing alignment loss:
# use the pretrained cls_token as the target to ensure that
# 1.all modality cls_token does not deviate.
# 2.the cls_token of each missing modality combinations being aligned
class alignMissingModalityLoss(torch.nn.Module):
    def __init__(self, temperature=1.0, alpha=10.0):
        super(alignMissingModalityLoss, self).__init__()
        self.temperature = temperature
        self.lsoftmax = nn.LogSoftmax(dim=0)
        self.alpha = alpha

    def forward(self, now_token_all, now_token_miss, target_token):
        mse = nn.MSELoss()
        loss1 = mse(now_token_all, target_token)
        loss2 = mse(now_token_miss, target_token.unsqueeze(0).expand_as(now_token_miss))
        loss = self.alpha * loss1 + loss2
        return loss

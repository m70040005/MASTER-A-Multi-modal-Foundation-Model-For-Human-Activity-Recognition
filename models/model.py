from models.feature_extraction import feature_extraction
from models.embedding import Embedding
from models.self_learning import self_learning
from models.recoverlayer import recoverlayer
from models.predictor import predictor
from utils import set_requires_grad
import torch.nn as nn


class model(nn.Module):
    '''
    A model class Used to integrate various module parts. Don't directly call it. Try this:\n
    model.feature_extracting_model() \n
    model.embedding_model() \n
    model.self_learning_model() \n
    output = model.output_model()
    '''
    def __init__(self,configs, training_mode, device):
        super(model, self).__init__()
        self.feature_extracting_model = feature_extraction(configs).to(device)
        self.embedding_model = Embedding(configs).to(device)
        self.self_learning_model = self_learning(configs, device).to(device)
        if training_mode == "s":
            self.output_model = recoverlayer(configs).to(device)
        elif training_mode == "a" or training_mode == "am" or training_mode == "f":
            self.output_model = predictor(configs).to(device)

    def load_parameters(self, chkpoint, layer_num=3, drop_last_layer=False):
        # Load parameter from checkpoint dict
        if layer_num >= 1:
            feature_extracting_dict = chkpoint["feature_extracting_dict"]
            temp_dict = self.feature_extracting_model.state_dict()
            temp_dict.update(feature_extracting_dict)
            self.feature_extracting_model.load_state_dict(temp_dict)
            set_requires_grad(self.feature_extracting_model, temp_dict, requires_grad=False)
        if layer_num >= 2:
            embedding_dict = chkpoint["embedding_dict"]
            temp_dict = self.embedding_model.state_dict()
            temp_dict.update(embedding_dict)
            self.embedding_model.load_state_dict(temp_dict)
            set_requires_grad(self.embedding_model, temp_dict, requires_grad=False)
        if layer_num >= 3:
            self_learning_dict = chkpoint["self_learning_dict"]
            temp_dict = self.self_learning_model.state_dict()
            temp_dict.update(self_learning_dict)
            self.self_learning_model.load_state_dict(temp_dict)
            set_requires_grad(self.self_learning_model, temp_dict, requires_grad=False)
        if layer_num >= 4:
            output_dict = chkpoint["output_dict"]
            if drop_last_layer:
                output_dict_copy = output_dict.copy()
                for key in output_dict.keys():
                    if "predictlayer" in key:
                        output_dict_copy.pop(key)
                output_dict = output_dict_copy
            temp_dict = self.output_model.state_dict()
            temp_dict.update(output_dict)
            self.output_model.load_state_dict(temp_dict)
            set_requires_grad(self.output_model, temp_dict, requires_grad=False)

    def model_set_requires_grad(self, model_choose, bool=True):
        if '1' in model_choose:
            temp_dict = self.feature_extracting_model.state_dict()
            set_requires_grad(self.feature_extracting_model, temp_dict, requires_grad=bool)
        if '2' in model_choose:
            temp_dict = self.embedding_model.state_dict()
            set_requires_grad(self.embedding_model, temp_dict, requires_grad=bool)
        if '3' in model_choose:
            temp_dict = self.self_learning_model.state_dict()
            set_requires_grad(self.self_learning_model, temp_dict, requires_grad=bool)
        if '4' in model_choose:
            temp_dict = self.output_model.state_dict()
            set_requires_grad(self.output_model, temp_dict, requires_grad=bool)


    def forward(self, x):

        return x

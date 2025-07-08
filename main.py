import torch
import os
import random
import numpy as np
from datetime import datetime
import argparse

from dataloader import data_generator
from trainer import Trainer, model_evaluate
from utils import _calc_metrics, copy_Files, _logger
from models.model import model


start_time = datetime.now()
##### Get running parameters #####
parser = argparse.ArgumentParser()
home_dir = os.getcwd()
# Training mode and dataset selection settings
parser.add_argument('--training_mode', default='s', type=str,
                    help='Modes of training. s: self_supervised, a: align_learning, am:align_miss_modality, f: fine_tune')
parser.add_argument('--selected_dataset', default='uci', type=str,
                    help='Set a dataset file name to find data and config')
parser.add_argument('--label_rate', default=0.2,type=float,
                    help='Data label rate, 0.2 means 20% of data is labeled and 80% of data is unlabeled')
parser.add_argument('--seed', default=123, type=int,
                    help='Seed value')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--cuda_no', default='0', type=str,
                    help='The no. of cuda use to train the model')
# Log storage path settings
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='Saving directory of logs(folder1)')
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description(folder2)')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Run Description(folder3)')

# An example of running parameters :
#python main.py --training_mode s --selected_dataset uci --label_rate 0.2 --seed 123 --device cuda --cuda_no 0 --experiment_description test --run_description uci

args = parser.parse_args()

training_mode = args.training_mode
assert training_mode == 's' or training_mode == 'a' or training_mode == 'am' or training_mode == 'f', \
    "Parameter error: Training mode is not in [s,a,am,f]"
dataset_name = args.selected_dataset
label_rate = args.label_rate
device = torch.device(args.device)
if device == torch.device('cuda'):
    cuda_no = args.cuda_no
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_no
logs_save_dir = args.logs_save_dir
experiment_description = args.experiment_description
run_description = args.run_description

##### Read config file #####
exec(f'from config_files.{dataset_name}_Configs import Config as Configs')
configs = Configs()

##### Fix random number seed #####
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

##### Generate log file directory(folder4) #####
os.makedirs(logs_save_dir, exist_ok=True)
if training_mode == "s":
    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, "self_supervised")
if training_mode == "a":
    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, f"align_learning")
if training_mode == 'am':
    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, f"align_miss_modality")
if training_mode == "f":
    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, "fine_tune" +
                                      f"_{label_rate}_time_{datetime.now().strftime('%m_%d_%H_%M_%S')}")
os.makedirs(experiment_log_dir, exist_ok=True)

##### Prepare for writing logs #####
log_file_name = os.path.join(experiment_log_dir, f"logs.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Mode:        {training_mode}')
logger.debug(f'Dataset:     {dataset_name}')
logger.debug(f'Label_rate:  {label_rate}')
if training_mode == "s":
    logger.debug(f'self_supervised_epoch   {configs.num_epoch}')
elif training_mode == "a":
    logger.debug(f'align_learning_epoch   {configs.num_epoch}')
elif training_mode == "am":
    logger.debug(f'align_miss_modality   {configs.num_epoch}')
elif training_mode == "f":
    logger.debug(f'fine_tune_epoch   {configs.num_epoch}_{configs.early_stop_step}')
logger.debug("=" * 45)

##### Load dataset #####
data_path = f"./data/{dataset_name}"
train_dl, test_dl = data_generator(data_path, configs, logger, training_mode, label_rate)

##### Load mode and saved model parameters #####
model = model(configs, training_mode, device)
model.model_set_requires_grad('1234',True)

if training_mode == "a":
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description,"self_supervised"))
    chkpoint = torch.load(os.path.join(load_from, "checkpoint.pt"), map_location=device)
    model.load_parameters(chkpoint, 3)
    model.model_set_requires_grad('123', False)
    model.model_set_requires_grad('4', True)
if training_mode == "am":
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description,"align_learning"))
    chkpoint = torch.load(os.path.join(load_from, "checkpoint.pt"), map_location=device)
    model.load_parameters(chkpoint, 4)
    model.model_set_requires_grad('12', False)
    model.model_set_requires_grad('34', True)
if training_mode == "f":
    load_from = os.path.join(logs_save_dir, experiment_description, run_description)
    if os.path.exists(os.path.join(load_from, f"align_miss_modality", "checkpoint.pt")):
        chkpoint = torch.load(os.path.join(load_from, f"align_miss_modality", "checkpoint.pt"), map_location=device)
        model.load_parameters(chkpoint, 4, drop_last_layer=True)
        model.model_set_requires_grad('4', True)
        model.model_set_requires_grad('123', False)
    elif os.path.exists(os.path.join(load_from, f"align_learning", "checkpoint.pt")):
        chkpoint = torch.load(os.path.join(load_from, f"align_learning", "checkpoint.pt"), map_location=device)
        model.load_parameters(chkpoint, 4, drop_last_layer=True)
        model.model_set_requires_grad('4', True)
        model.model_set_requires_grad('123', False)
    elif os.path.exists(os.path.join(load_from, "self_supervised", "checkpoint.pt")):
        chkpoint = torch.load(os.path.join(load_from, "self_supervised", "checkpoint.pt"), map_location=device)
        model.load_parameters(chkpoint, 3, drop_last_layer=True)
        model.model_set_requires_grad('4', True)
        model.model_set_requires_grad('123', False)
    else:
        logger.debug(f"Warning: self_supervised weights not found, start fine_tune with init model parameters")
        model.model_set_requires_grad('1234', True)

##### Code backup #####
if training_mode == "s":
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), dataset_name)

##### Train #####
Trainer(model, train_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode)

if training_mode == "f":
    total_loss, total_acc, pred_labels, true_labels = model_evaluate(model, test_dl, device, training_mode)
    _calc_metrics(pred_labels, true_labels, experiment_log_dir)

logger.debug(f"Training time is : {datetime.now()-start_time}")

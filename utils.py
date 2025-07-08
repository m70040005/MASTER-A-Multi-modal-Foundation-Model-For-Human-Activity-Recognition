import torch
import numpy as np
import pandas as pd
import os
import sys
import logging
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
from shutil import copy, copytree, rmtree


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad


def _calc_metrics(pred_labels, true_labels, log_dir):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    os.makedirs(log_dir, exist_ok=True)
    result_dict = dict()
    result_dict["predict labels"] = pred_labels
    result_dict["true labels"] = true_labels
    result_dict["equal"] = true_labels.__eq__(pred_labels).astype(int)
    df = pd.DataFrame(result_dict)
    df.to_excel(os.path.join(log_dir, "results.xlsx"))

    np.save(os.path.join(log_dir, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(log_dir, "true_labels.npy"), true_labels)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df = df * 100

    # save classification report
    exp_name = os.path.split(os.path.dirname(log_dir))[-1]
    training_mode = os.path.basename(log_dir)
    file_name = f"{exp_name}_{training_mode}_classification_report.xlsx"
    report_Save_path = os.path.join(log_dir, file_name)
    df.to_excel(report_Save_path)

    # save confusion matrix
    cm_file_name = f"{exp_name}_{training_mode}_confusion_matrix.torch"
    cm_Save_path = os.path.join(log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def copy_Files(destination, dataset_name):
    destination_dir = os.path.join(destination, "model_files")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("trainer.py", os.path.join(destination_dir, "trainer.py"))
    copy("dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"config_files/{dataset_name}_Configs.py", os.path.join(destination_dir, f"{dataset_name}_Configs.py"))
    if os.path.exists(os.path.join(destination_dir, "modality_process")):
        rmtree(os.path.join(destination_dir, "modality_process"))
    copytree("modality_process", os.path.join(destination_dir, "modality_process"))
    if os.path.exists(os.path.join(destination_dir, f"models")):
        rmtree(os.path.join(destination_dir, f"models"))
    copytree("models", os.path.join(destination_dir, f"models"))

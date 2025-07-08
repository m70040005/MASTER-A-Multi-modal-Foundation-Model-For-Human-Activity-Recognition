import os
import sys
import numpy as np
import itertools
from tqdm import tqdm
import torch
import torch.nn as nn
from models.train_loss import selfLearningLoss, alignLearningLoss, alignMissingModalityLoss
from utils import AverageMeter
import copy
sys.path.append("")


def Trainer(model, train_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode):
    ######################### Start training  #########################
    logger.debug("Training started ....")

    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min', factor=0.1, cooldown=5, patience=10)

    acc_max = 0
    count1 = 0  # early stop counter
    count2 = 0  # freeze exchange counter
    freeze = True  # record which part is frozen

    if training_mode == "am":
        model_copy = copy.deepcopy(model)
        model_copy.model_set_requires_grad('1234', False)
        model_copy.eval()
    else:
        model_copy = None

    for epoch in range(1, configs.num_epoch + 1):
        ######################### Training #########################
        train_loss, train_acc = model_train(model, model_copy, model_optimizer, train_dl, device, training_mode, configs)
        valid_loss, valid_acc, _, _ = model_evaluate(model, test_dl, device, training_mode)

        lr = model_optimizer.param_groups[0]['lr']
        logger.debug(f'\nEpoch : {epoch}  Learning Rate :    {lr}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:4.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:4.4f}')

        scheduler.step(train_loss)
        if lr < 1e-8:
            logger.debug("\n################## Early stop! #########################")
            break

        count2 += 1
        if training_mode == "s":
            ######################### Save parameter in self supervised, align learning and align_miss_modality mode #########################
            chkpoint = {'feature_extracting_dict': model.feature_extracting_model.state_dict(),
                        'embedding_dict': model.embedding_model.state_dict(),
                        'self_learning_dict': model.self_learning_model.state_dict(),
                        'output_dict': model.output_model.state_dict()}
            torch.save(chkpoint, os.path.join(experiment_log_dir, 'checkpoint.pt'))
            if (epoch % 10 == 0):
                chkpoint = {'feature_extracting_dict': model.feature_extracting_model.state_dict(),
                            'embedding_dict': model.embedding_model.state_dict(),
                            'self_learning_dict': model.self_learning_model.state_dict(),
                            'output_dict': model.output_model.state_dict()}
                torch.save(chkpoint, os.path.join(experiment_log_dir, 'checkpoint' + str(epoch) + '.pt'))
        elif training_mode == "a":
            chkpoint = {'feature_extracting_dict': model.feature_extracting_model.state_dict(),
                        'embedding_dict': model.embedding_model.state_dict(),
                        'self_learning_dict': model.self_learning_model.state_dict(),
                        'output_dict': model.output_model.state_dict()}
            torch.save(chkpoint, os.path.join(experiment_log_dir, 'checkpoint.pt'))
            if (epoch % 10 == 0):
                chkpoint = {'feature_extracting_dict': model.feature_extracting_model.state_dict(),
                            'embedding_dict': model.embedding_model.state_dict(),
                            'self_learning_dict': model.self_learning_model.state_dict(),
                            'output_dict': model.output_model.state_dict()}
                torch.save(chkpoint, os.path.join(experiment_log_dir, 'checkpoint' + str(epoch) + '.pt'))
        elif training_mode == "am":
            chkpoint = {'feature_extracting_dict': model.feature_extracting_model.state_dict(),
                        'embedding_dict': model.embedding_model.state_dict(),
                        'self_learning_dict': model.self_learning_model.state_dict(),
                        'output_dict': model.output_model.state_dict()}
            torch.save(chkpoint, os.path.join(experiment_log_dir, 'checkpoint.pt'))
            if (epoch % 5 == 0):
                chkpoint = {'feature_extracting_dict': model.feature_extracting_model.state_dict(),
                            'embedding_dict': model.embedding_model.state_dict(),
                            'self_learning_dict': model.self_learning_model.state_dict(),
                            'output_dict': model.output_model.state_dict()}
                torch.save(chkpoint, os.path.join(experiment_log_dir, 'checkpoint' + str(epoch) + '.pt'))
        elif training_mode == "f":
            ######################### Change the frozen part of the model after several epoch #########################
            if freeze == True:
                if count2 >= configs.num_exchange_freeze_epoch:
                    model.model_set_requires_grad('4', False)
                    model.model_set_requires_grad('123', True)
                    count2 = 0
                    freeze = False
            else:
                if count2 >= configs.num_exchange_unfreeze_epoch:
                    model.model_set_requires_grad('4', True)
                    model.model_set_requires_grad('123', False)
                    count2 = 0
                    freeze = True
            ######################### Save best parameter #########################
            if valid_acc > acc_max:
                acc_max = valid_acc
                count1 = 0
                chkpoint = {'feature_extracting_dict': model.feature_extracting_model.state_dict(),
                            'embedding_dict': model.embedding_model.state_dict(),
                            'self_learning_dict': model.self_learning_model.state_dict(),
                            'output_dict': model.output_model.state_dict()}
                torch.save(chkpoint, os.path.join(experiment_log_dir, 'checkpoint.pt'))
                logger.debug("-------------------- Best update! --------------------")
            else:
                count1 += 1
                ######################### If best ACC is not updated multiple times #########################
                if count1 >= configs.early_stop_step:
                    logger.debug("\n################## Early stop! #########################")
                    break

    ######################### Testing #########################
    if training_mode == "f":
        ######################## load best parameter #########################
        if configs.num_epoch > 0:
            chkpoint = torch.load(os.path.join(experiment_log_dir, "checkpoint.pt"), map_location=device)
            model.load_parameters(chkpoint, 4)

        logger.debug('\nEvaluate on the Test set:')
        model_evaluate_final(model, test_dl, device, training_mode, configs.missing_modality_selection_range, logger, modality_missing_num=0)

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, model_copy, model_optimizer, train_loader, device, training_mode, configs):
    total_loss = AverageMeter()
    total_acc = AverageMeter()
    model.train()

    criterion = nn.CrossEntropyLoss()

    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for idx, (data, labels, length_mask, dfrom) in loop:
        batch_length = labels.shape[0]
        ######################### Move tensors #########################
        for key in data.keys():
            data[key] = data[key].float().to(device)
            length_mask[key] = length_mask[key].to(device)
        labels = labels.long().to(device)

        if training_mode == "am":
            data_miss_copy = copy.deepcopy(data)
            length_mask_miss_copy = copy.deepcopy(length_mask)
            data_old = copy.deepcopy(data)
            length_mask_old = copy.deepcopy(length_mask)

        ######################### Model training #########################
        model.zero_grad()

        features, length_mask = model.feature_extracting_model(data, length_mask)
        embedding_features, cls_token = model.embedding_model(features, length_mask)
        forward_seq, target_features, loss_mask_seq, output_token = model.self_learning_model(embedding_features,length_mask,cls_token,training_mode)
        if training_mode == "s":
            output = model.output_model(forward_seq)
        elif training_mode == "a":
            output = model.output_model.feature_proj_block(output_token)
        elif training_mode == "am":
            modality_list = configs.missing_modality_selection_range
            miss_modality_list = []
            for num in range(1, len(modality_list)):
                for mlist in itertools.combinations(modality_list, num):
                    final_list = []
                    for i in range(num):
                        final_list = final_list + mlist[i]
                    miss_modality_list.append(final_list)
            output_token_miss_list = []
            for slist in miss_modality_list:
                data_miss = copy.deepcopy(data_miss_copy)
                length_mask_miss = copy.deepcopy(length_mask_miss_copy)
                for key in slist:
                    data_miss[key] = torch.zeros_like(data_miss[key]).float().to(device)
                    length_mask_miss[key] = torch.zeros_like(length_mask_miss[key]).to(device)

                features_miss, length_mask_miss = model.feature_extracting_model(data_miss, length_mask_miss)
                embedding_features_miss, length_mask_miss, cls_token_miss = model.embedding_model(features_miss, length_mask_miss)
                _, _, _, output_token_miss, _ = model.self_learning_model(embedding_features_miss, length_mask_miss,
                                                                          cls_token_miss, training_mode)
                output_token_miss_list.append(output_token_miss.unsqueeze(0))
            output_token_miss_all = torch.cat(output_token_miss_list, dim=0)

            features_old, length_mask_old = model_copy.feature_extracting_model(data_old, length_mask_old)
            embedding_features_old, length_mask_old, cls_token_old = model_copy.embedding_model(features_old, length_mask_old)
            _, _, _, output_token_old, _ = model_copy.self_learning_model(embedding_features_old, length_mask_old, cls_token_old, training_mode)
        elif training_mode == "f":
            output = model.output_model(output_token)


        ######################### Calculate loss #########################
        if training_mode == "s":
            infoNCE_loss = selfLearningLoss(temperature=configs.temperature)
            loss = infoNCE_loss(output, target_features, loss_mask_seq)
        elif training_mode == "a":
            alignLearning_loss = alignLearningLoss(temperature=configs.temperature)
            loss = alignLearning_loss(output, labels)
        elif training_mode == "am":
            alignLearning_loss = alignMissingModalityLoss(temperature=configs.temperature)
            loss = alignLearning_loss(output_token, output_token_miss_all, output_token_old)
        elif training_mode == "f":
            loss = criterion(output.squeeze(), labels)
            total_acc.update(sum(labels.eq(output.detach().argmax(dim=1)).float())/batch_length, batch_length)
        total_loss.update(loss.item(), batch_length)
        loss.backward()
        model_optimizer.step()

    if training_mode == "s" or training_mode == "a" or training_mode == "am":
        total_acc.reset()

    return total_loss.avg, total_acc.avg


def model_evaluate(model, test_dl, device, training_mode):
    model.eval()

    total_loss = AverageMeter()
    total_acc = AverageMeter()

    criterion = nn.CrossEntropyLoss()

    outs = np.array([])
    trgs = np.array([])

    if training_mode == "s" or training_mode == "a" or training_mode == "am":  # Self supervised, align learning and align_miss_modality mode don't need test
        return 0, 0, [], []

    loop = tqdm(enumerate(test_dl), total=len(test_dl))

    with torch.no_grad():
        for idx, (data, labels, length_mask, dfrom) in loop:
            batch_length = labels.shape[0]
            ######################### Move tensors #########################
            for key in data.keys():
                data[key] = data[key].float().to(device)
                length_mask[key] = length_mask[key].to(device)
            labels = labels.long().to(device)


            features, length_mask = model.feature_extracting_model(data, length_mask)
            embedding_features, cls_token = model.embedding_model(features, length_mask)
            _, _, _, output_token = model.self_learning_model(embedding_features, length_mask, cls_token, training_mode)
            if training_mode == "f":
                output = model.output_model(output_token)

            ######################### Calculate loss, record predictions #########################
            if training_mode == "f":
                loss = criterion(output.squeeze(), labels)
                total_acc.update(sum(labels.eq(output.detach().argmax(dim=1)).float())/batch_length, batch_length)
                total_loss.update(loss.item(), batch_length)

                pred = output.max(1, keepdim=True)[1]
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    return total_loss.avg, total_acc.avg, outs, trgs

def model_evaluate_final(model, test_dl, device, training_mode, modality_selection_range, logger, modality_missing_num):
    model.eval()

    if training_mode == "s" or training_mode == "a" or training_mode == "am":  # Self supervised, align learning and align_miss_modality mode don't need test
        return 0, 0


    missing_modality = [[]]
    modality = modality_selection_range # if you want to use missing modality evaluation, you need to set modality selection range
    if modality_missing_num != 0:
        for list in itertools.combinations(modality, modality_missing_num):
            final_list = []
            for i in range(modality_missing_num):
                final_list = final_list + list[i]
            missing_modality.append(final_list)

    test_num = len(missing_modality)
    test_acc_of_different_set = []
    for test_i in range(test_num):
        total_loss = AverageMeter()
        total_acc = AverageMeter()

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data_raw, labels, length_mask_raw, dfrom in test_dl:
                batch_length = labels.shape[0]
                ######################### Move tensors #########################
                data = dict()
                length_mask = dict()
                for key in data_raw.keys():
                    data[key] = data_raw[key].float().to(device)
                    length_mask[key] = length_mask_raw[key].to(device)
                    if key in missing_modality[test_i]:
                        data[key] = torch.zeros_like(data[key]).float().to(device)
                        length_mask[key] = torch.zeros_like(length_mask[key]).to(device)

                labels = labels.long().to(device)

                features, length_mask = model.feature_extracting_model(data, length_mask)
                embedding_features, cls_token = model.embedding_model(features, length_mask)
                _, _, _, output_token = model.self_learning_model(embedding_features, length_mask,
                                                                               cls_token, training_mode)
                if training_mode == "f":
                    output = model.output_model(output_token)

                ######################### Calculate loss, record predictions #########################
                if training_mode == "f":
                    loss = criterion(output.squeeze(), labels)
                    total_acc.update(sum(labels.eq(output.detach().argmax(dim=1)).float())/batch_length, batch_length)
                    total_loss.update(loss.item(), batch_length)

        logger.debug(f'\ntesting test_loder {test_i}: ')
        logger.debug(f'\nmissing modalitys: {missing_modality[test_i]}: ')
        logger.debug(f'\nTest loss      :{total_loss.avg:0.4f}\t | Test Accuracy      : {total_acc.avg:0.4f}')
        if test_i != 0:
            test_acc_of_different_set.append(total_acc.avg.cpu())

    logger.debug(f'\naverage test Accuracy(except all modality test)                : {np.mean(np.array(test_acc_of_different_set)):0.4f}')
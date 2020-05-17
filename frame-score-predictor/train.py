from utils.arguments import parse_arguments
from utils.dataset import COVID19Dataset
from utils.tranforms import get_transforms
import torch.optim as optim
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision.utils import save_image
import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models

from skimage import io, img_as_float32
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import os
import cv2
import imageio
import colorama
from colorama import Fore, Back, Style
colorama.init()
from tqdm import tqdm, trange
from time import gmtime, strftime
import pandas as pd
from random import randint

from models.network import CNNConStn

def sord_loss(logits, ground_truth, num_classes=4, multiplier=2, wide_gap_loss=False):
    batch_size = ground_truth.size(0)
    # Allocates sord probability vector
    labels_sord = np.zeros((batch_size, num_classes))
    for element_idx in range(batch_size):
        current_label = ground_truth[element_idx].item()
        # Foreach class compute the distance between ground truth label and current class
        for class_idx in range(num_classes):
            # Distance computation that increases the distance between negative patients and
            # positive patients in the sord loss.
            if wide_gap_loss:
                wide_label = current_label
                wide_class_idx = class_idx
                # Increases the gap between positive and negative
                if wide_label == 0:
                    wide_label = -0.5
                if wide_class_idx == 0:
                    wide_class_idx = -0.5
                labels_sord[element_idx][class_idx] = multiplier * abs(wide_label - wide_class_idx) ** 2
            # Standard computation distance = 2 * ((class label - ground truth label))^2
            else:
                labels_sord[element_idx][class_idx] = multiplier * abs(current_label - class_idx) ** 2
    labels_sord = torch.from_numpy(labels_sord).cuda(non_blocking=True)
    labels_sord = F.softmax(-labels_sord, dim=1)
    # Uses log softmax for numerical stability
    log_predictions = F.log_softmax(logits, 1)
    # Computes cross entropy
    loss = (-labels_sord * log_predictions).sum(dim=1).mean()
    return loss

def train(args, model, train_loader, nclasses, optimizer, epoch, fixed_samples, fixed_y):
    model.train()
    correct = 0
    train_loss, mse_losses, stn_reg_losses, preds, labels = [], [], [], [], []
    confusion_matrix = torch.zeros(nclasses, nclasses)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.long().cuda()
        output, scaling = model(data)
        optimizer.zero_grad()
        output_1, output_2 = torch.split(output, split_size_or_sections=output.shape[0] // 2)
        output = output_1

        # scaling loss
        if not args.fixed_scale:
            stn_reg_loss = args.lambda_stn_params * nn.L1Loss()(
                torch.tensor([0.5, 0.75]).view(1, 2).repeat(data.shape[0], 1).cuda(), scaling
            )
            stn_reg_losses.append(stn_reg_loss.item())

        # supervised loss
        loss = sord_loss(logits=output, ground_truth=target, num_classes=nclasses, multiplier=args.multiplier)
        train_loss.append(loss.item())

        # consistency loss
        mse_loss = args.lambda_cons * torch.pow((output_1 - output_2), 2).mean()
        mse_losses.append(mse_loss.item())

        if not args.fixed_scale:
            (loss + mse_loss + stn_reg_loss).backward()
        else:
            (loss + mse_loss).backward() # for translation only
        optimizer.step()
        pred = F.softmax(output, dim=1).max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        # to compute metrics
        preds.append(pred.view(-1).cpu())
        labels.append(target.view(-1).cpu())
        
        for t, p in zip(target.view(-1), pred.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        if not args.fixed_scale:
            if batch_idx % args.log_interval == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tCELoss: {:.6f}\tConLoss: {:.6f}\tScalingLoss: {:.6f}'.format(epoch,
                                                                                          batch_idx * len(data),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item(), mse_loss.item(), stn_reg_loss.item()))
        else:
            if batch_idx % args.log_interval == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tCELoss: {:.6f}\tConLoss: {:.6f}'.format(epoch,
                                                                                          batch_idx * len(data),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item(), mse_loss.item()))

    # compute the metrics
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true=torch.cat(labels), y_pred=torch.cat(preds), average='micro')

    # print the logs
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    print(Fore.GREEN + '\nTrain set: Accuracy: {}/{}({:.2f}%)'.format(correct,
    len(train_loader.dataset), 100 * correct / len(train_loader.dataset)) +
    Style.RESET_ALL)

    print(Fore.GREEN + 'Classwise Accuracy:: Cl-0: {}/{}({:.2f}%),\
    Cl-1: {}/{}({:.2f}%), Cl-2: {}/{}({:.2f}%), Cl-3: {}/{}({:.2f}%); \
    Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(
    int(confusion_matrix.diag()[0].item()), int(confusion_matrix.sum(1)[0].item()), per_class_accuracy[0].item() * 100.,
    int(confusion_matrix.diag()[1].item()), int(confusion_matrix.sum(1)[1].item()), per_class_accuracy[1].item() * 100.,
    int(confusion_matrix.diag()[2].item()), int(confusion_matrix.sum(1)[2].item()), per_class_accuracy[2].item() * 100.,
    int(confusion_matrix.diag()[3].item()), int(confusion_matrix.sum(1)[3].item()), per_class_accuracy[3].item() * 100.,
    precision, recall, fscore) + Style.RESET_ALL)

    # visualize the transformations
    filler = torch.ones(1, 3, args.img_size, args.img_size)
    fixed_samples = torch.cat([fixed_samples[fixed_y == 0, ...],
                               filler,
                               fixed_samples[fixed_y == 1, ...],
                               filler,
                               fixed_samples[fixed_y == 2, ...],
                               filler,
                               fixed_samples[fixed_y == 3, ...]], dim=0)
    stn_out = F.interpolate(model.stn(fixed_samples.cuda())[0].cpu(), size=(args.img_size, args.img_size))
    stn_out_1, stn_out_2 = torch.split(stn_out, split_size_or_sections=stn_out.shape[0] // 2)
    viz_tensor = torch.cat([fixed_samples, stn_out_1, stn_out_2], dim=3)
    save_image(viz_tensor, os.path.join(args.train_viz_dir, str(epoch).zfill(4) + '.png'), nrow=int(viz_tensor.shape[0] ** 0.5))

    return model

def test(args, model, test_loader, nclasses, epoch, state_dict, weights_path, fixed_samples):
    model.eval()
    test_losses = []
    correct = 0
    preds, labels = [], []
    confusion_matrix = torch.zeros(nclasses, nclasses)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.long().cuda()
            output, _ = model(data)
            output_1, output_2 = torch.split(output, split_size_or_sections=output.shape[0] // 2)
            output = output_1
            loss = sord_loss(logits=output, ground_truth=target, num_classes=nclasses, multiplier=args.multiplier)
            test_losses.append(loss.item())
            pred = F.softmax(output, dim=1).max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            # compute metrics
            preds.append(pred.view(-1).cpu())
            labels.append(target.view(-1).cpu())

            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    test_loss = np.mean(np.asarray(test_losses))

    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true=torch.cat(labels), y_pred=torch.cat(preds), average='micro')
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    print(Fore.RED + '\nTest Set: Average Loss: {:.4f}, Accuracy: {}/{} \
    ({:.2f}%)'.format(test_loss, correct, len(test_loader.dataset), 100 *
    correct / len(test_loader.dataset))  + Style.RESET_ALL)

    print(Fore.RED + 'Classwise Accuracy:: Cl-0: {}/{}({:.2f}%), Cl-1: {}/{}({:.2f}%) \
    Cl-2: {}/{}({:.2f}%), Cl-3: {}/{}({:.2f}%); \
    Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(
    int(confusion_matrix.diag()[0].item()), int(confusion_matrix.sum(1)[0].item()), per_class_accuracy[0].item() * 100.,
    int(confusion_matrix.diag()[1].item()), int(confusion_matrix.sum(1)[1].item()), per_class_accuracy[1].item() * 100.,
    int(confusion_matrix.diag()[2].item()), int(confusion_matrix.sum(1)[2].item()), per_class_accuracy[2].item() * 100.,
    int(confusion_matrix.diag()[3].item()), int(confusion_matrix.sum(1)[3].item()), per_class_accuracy[3].item() * 100.,
    precision, recall, fscore) + Style.RESET_ALL)

    metrics = {'test/accuracy': correct / len(test_loader.dataset) * 100.,
              'test/precision': precision,
              'test/recall': recall,
              'test/F1': fscore,
              'test/loss': test_loss}

    print('Saving weights...')
    save_weights(model, os.path.join(weights_path, 'model.pth'))
    save_best_model(model, weights_path, metrics, state_dict)

    # visualize the transformations
    stn_out = F.interpolate(model.stn(fixed_samples.cuda())[0].cpu(), size=(args.img_size, args.img_size))
    stn_out_1, stn_out_2 = torch.split(stn_out, split_size_or_sections=stn_out.shape[0] // 2)
    viz_tensor = torch.cat([fixed_samples, stn_out_1, stn_out_2], dim=3)
    save_image(viz_tensor, os.path.join(args.test_viz_dir, str(epoch).zfill(4) + '.png'), nrow=int(viz_tensor.shape[0] ** 0.5))

def get_weights_for_balanced_classes(labels, nclasses):
    count = [0] * nclasses
    for item in labels:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        if count[i] != 0:
            weight_per_class[i] = N / float(count[i])
    weight = [0] * len(labels)
    for idx, val in enumerate(labels):
        weight[idx] = weight_per_class[val]
    return weight

def save_weights(model, path):
    torch.save(model.state_dict(), path)

def load_weights(args, model, path):
    if args.arch == 'ResNet50':
        state_dict_ = torch.load(path)
        modified_state_dict = {}
        for key in state_dict_.keys():
            mod_key = key[7:]
            modified_state_dict.update({mod_key: state_dict_[key]})
    else:
        modified_state_dict = torch.load(path)
    model.load_state_dict(modified_state_dict, strict=True)
    return model

def save_best_model(model, path, metrics, state_dict):
    if metrics['test/F1'] > state_dict['best_f1']:
        state_dict['best_f1'] = max(metrics['test/F1'], state_dict['best_f1'])
        state_dict['accuracy'] = metrics['test/accuracy']
        state_dict['precision'] = metrics['test/precision']
        state_dict['recall'] = metrics['test/recall']
        print('F1 score improved over the previous. Saving model...')
        save_weights(model=model, path=os.path.join(path, 'best_model.pth'))
    best_str = "Best Metrics:" + '; '.join(["%s - %s" % (k, v) for k, v in state_dict.items()])
    print(Fore.BLUE + best_str + Style.RESET_ALL)


def experiment(args):
    # load data
    data = pd.read_pickle(os.path.join(args.dataset_root, 'dataset.pkl'))
    data = data[data.sensor.str.contains('|'.join(args.sensors))]  # filter sensors

    # load splits
    splits = pd.read_csv(os.path.join(args.dataset_root, 'train_test_split.csv'))
    train_patients = splits[splits.split.str.contains('train')].patient_hash.tolist()
    test_patients = splits[splits.split.str.contains('test')].patient_hash.tolist()

    # get data accorting to patient split
    train_data = data[data.patient_hash.str.contains('|'.join(train_patients))]
    test_data = data[data.patient_hash.str.contains('|'.join(test_patients))]

    # subset the dataset
    train_dataset = COVID19Dataset(args, train_data, get_transforms(args, 'train'))
    test_dataset = COVID19Dataset(args, test_data, get_transforms(args, 'test'))

    # For unbalanced dataset we create a weighted sampler
    train_labels = [sum(l) for l in train_data.label.tolist()]
    weights = get_weights_for_balanced_classes(train_labels, len(list(set(train_labels))))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights, num_samples=len(weights))

    nclasses = len(list(set(train_labels)))
    # dataloaders from subsets
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=args.num_workers,
        drop_last=False)

    # create directories
    args.weights_dir = os.path.join('logs', args.run_name, 'weights')
    os.makedirs(args.weights_dir, exist_ok=True)
    args.train_viz_dir = os.path.join('logs', args.run_name, 'viz_train')
    os.makedirs(args.train_viz_dir, exist_ok=True)
    args.test_viz_dir = os.path.join('logs', args.run_name, 'viz_test')
    os.makedirs(args.test_viz_dir, exist_ok=True)

    model = CNNConStn(args.img_size, nclasses, args.fixed_scale)
    print(model)
    print('Number of params in the model: {}'.format(
        *[sum([p.data.nelement() for p in net.parameters()]) for net in [model]]))
    model = model.cuda()

    # fixed samples for stn visualization
    fixed_samples_iter = iter(train_loader)
    fixed_samples_train, fixed_y_train = fixed_samples_iter.next()
    fixed_samples_iter = iter(test_loader)
    fixed_samples_test, _ = fixed_samples_iter.next()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[70], gamma=0.1) # 10, 50
    state_dict = {'best_f1': 0., 'precision': 0., 'recall': 0., 'accuracy': 0.}
    for epoch in range(args.epochs):
        model = train(args, model, train_loader, len(list(set(train_labels))), optimizer, epoch,
                      fixed_samples_train, fixed_y_train)
        test(args, model, test_loader, len(list(set(train_labels))), epoch, state_dict, args.weights_dir,
             fixed_samples_test)
        exp_lr_scheduler.step()


if __name__ == '__main__':
    cudnn.benchmark = True
    args = parse_arguments()
    print(args)
    experiment(args)
    
    

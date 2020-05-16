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
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import wandb
from random import randint
from plotnine import *

from models.network import CNN2D, SimpleCNN, WideResnet, WhitenMobileNetV2, CNNProj, CNNStn, CNNConStn

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

class MinEntropyConsensusLoss(nn.Module):
	def __init__(self, num_classes, device):
		super(MinEntropyConsensusLoss, self).__init__()
		self.num_classes = num_classes
		self.device = device

	def forward(self, x, y):
		i = torch.eye(self.num_classes, device=self.device).unsqueeze(0)
		x = F.log_softmax(x, dim=1)
		y = F.log_softmax(y, dim=1)

		x = x.unsqueeze(-1)
		y = y.unsqueeze(-1)

		ce_x = (- 1.0 * i * x).sum(1)
		ce_y = (- 1.0 * i * y).sum(1)

		ce = 0.5 * (ce_x + ce_y).min(1)[0].mean()

		return ce

def sample_theta(args, batch_size):
    noise = torch.normal(mean=0, std=args.sigma * torch.ones([batch_size, 2, 3]))
    theta = noise + torch.eye(2, 3).view(1, 2, 3)
    return theta

def affine_transform_images(args, imgs):
    theta = sample_theta(args, imgs.shape[0]).cuda()
    grid = F.affine_grid(theta, imgs.size())
    imgs = F.grid_sample(imgs, grid)
    return imgs

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def loss_criterion(nclasses, weights=None, smoothing=0.0):
    if smoothing != 0:
        criterion = LabelSmoothingLoss(classes=nclasses, smoothing=smoothing)
        return criterion
    #if weights is None:
        #weights = nclasses * [1.0]
        #weights = torch.FloatTensor(weights).cuda()
    #criterion = nn.CrossEntropyLoss(weight=weights, reduction='none')
    criterion = nn.CrossEntropyLoss(reduction='none')
    return criterion

def train(args, device, criterion, model, train_loader, nclasses, optimizer, epoch, wandb, fixed_samples, fixed_y):
    model.train()
    correct = 0
    train_loss, mse_losses, stn_reg_losses, preds, labels = [], [], [], [], []
    confusion_matrix = torch.zeros(nclasses, nclasses)
    for batch_idx, (data, data_duplicate, target, sensor, hospital) in enumerate(train_loader):
        data, data_duplicate, target, sensor, hospital = data.cuda(), data_duplicate.cuda(), target.long().cuda(), \
                                                         sensor.long().cuda(), \
                                         hospital.long().cuda()
        output, scaling = model(data, sensor) # for generic
        output_duplicate, _ = model(data_duplicate, sensor)  # for generic
        optimizer.zero_grad()
        if args.arch == 'CNNConStn':
            output_1, output_2 = torch.split(output, split_size_or_sections=output.shape[0] // 2)
            output = output_1
            if args.lambda_stn_params != 0:
                stn_reg_loss = args.lambda_stn_params * nn.L1Loss()(
                    torch.tensor([0.15, 0.30]).view(1, 2).repeat(data.shape[0], 1).cuda(), scaling ** 2 # [0.5, 0.75]
                )
                stn_reg_losses.append(stn_reg_loss.item())

        if args.multiplier != 0:
            loss = sord_loss(logits=output, ground_truth=target, num_classes=nclasses, multiplier=args.multiplier)
        else:
            loss = criterion(output, target).mean()
        train_loss.append(loss.item())

        if args.arch == 'CNNConStn':
            mse_loss = args.lambda_cons * torch.pow((output_1 - output_2), 2).mean()
            mse_losses.append(mse_loss.item())
        elif args.arch == 'CNNCon':
            mse_loss = args.lambda_cons * torch.pow((output - output_duplicate), 2).mean()
            mse_losses.append(mse_loss.item())

        # do the backward pass
        if args.arch in ['CNNStn', 'CNN2D']:
            loss.backward()
        elif args.arch in ['CNNConStn', 'CNNCon']:
            if args.lambda_cons != 0 and args.lambda_stn_params != 0:
                (loss + mse_loss + stn_reg_loss).backward() # for generic
            elif args.lambda_cons != 0:
                (loss + mse_loss).backward() # for translation only
        optimizer.step()
        pred = F.softmax(output, dim=1).max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        # to compute metrics
        preds.append(pred.view(-1).cpu())
        labels.append(target.view(-1).cpu())
        
        for t, p in zip(target.view(-1), pred.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        if args.arch in ['CNNStn', 'CNN2D']:
            if batch_idx % args.log_interval == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tCELoss: {:.6f}'.format(epoch, batch_idx * len(data),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
        elif args.arch in ['CNNConStn', 'CNNCon']:
            if args.lambda_stn_params != 0:
                if batch_idx % args.log_interval == 0:
                    print('Train epoch: {} [{}/{} ({:.0f}%)]\tCELoss: {:.6f}\tConLoss: {:.6f}\tBoxLoss: {:.6f}'.format(epoch,
                                                                                          batch_idx * len(data),
                    len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss.item(), mse_loss.item(), stn_reg_loss.item()))
            else:
                if batch_idx % args.log_interval == 0:
                    print('Train epoch: {} [{}/{} ({:.0f}%)]\tCELoss: {:.6f}\tConLoss: {:.6f}'.format(epoch,
                                                                                          batch_idx * len(data),
                    len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss.item(), mse_loss.item()))
    train_loss = np.mean(np.asarray(train_loss))
    mse_losses = np.mean(np.asarray(mse_losses))
    stn_reg_losses = np.mean(np.asarray(stn_reg_losses))
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true=torch.cat(labels), y_pred=torch.cat(preds), average='micro')

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

    wandb.log({'train/accuracy': correct / len(train_loader.dataset) * 100.,
               'train/precision': precision, 'train_recall': recall,
               'train/F1': fscore,
               'train/celoss': train_loss,
               'train/conloss': mse_losses,
               'train/stn_reg_loss': stn_reg_losses.item()}, step=epoch)

    for param_group in optimizer.param_groups:
        wandb.log({'lr': param_group['lr']}, step=epoch)

    # visualize the transformations
    if args.arch == 'CNNStn':
        filler = torch.ones(1, 3, args.img_size, args.img_size)
        fixed_samples = torch.cat([fixed_samples[fixed_y == 0, ...],
                                   filler,
                                   fixed_samples[fixed_y == 1, ...],
                                   filler,
                                   fixed_samples[fixed_y == 2, ...],
                                   filler,
                                   fixed_samples[fixed_y == 3, ...]], dim=0)
        stn_out = F.interpolate(model.stn(fixed_samples.cuda()).cpu(), size=(args.img_size, args.img_size))
        viz_tensor = torch.cat([fixed_samples, stn_out], dim=3)
        save_image(viz_tensor, os.path.join(args.train_viz_dir, str(epoch).zfill(4) + '.png'), nrow=int(viz_tensor.shape[0] ** 0.5))
    elif args.arch == 'CNNCon':
        filler = torch.ones(1, 3, args.img_size, args.img_size)
        a = fixed_samples['train']
        b = fixed_samples['train_dup']
        fixed_samples = torch.cat([a[fixed_y == 0, ...],
                                   filler,
                                   a[fixed_y == 1, ...],
                                   filler,
                                   a[fixed_y == 2, ...],
                                   filler,
                                   a[fixed_y == 3, ...]], dim=0)
        fixed_samples_dup = torch.cat([b[fixed_y == 0, ...],
                                   filler,
                                   b[fixed_y == 1, ...],
                                   filler,
                                   b[fixed_y == 2, ...],
                                   filler,
                                   b[fixed_y == 3, ...]], dim=0)
        viz_tensor = torch.cat([fixed_samples, fixed_samples_dup], dim=3)
        save_image(viz_tensor, os.path.join(args.train_viz_dir, str(epoch).zfill(4) + '.png'), nrow=int(viz_tensor.shape[0] ** 0.5))
    elif args.arch == 'CNNConStn':
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

def test(args, model, criterion, test_loader, nclasses, epoch, state_dict, weights_path, wandb, fixed_samples):
    model.eval()
    test_losses = []
    correct = 0
    preds, labels = [], []
    confusion_matrix = torch.zeros(nclasses, nclasses)
    with torch.no_grad():
        for data, target, sensor, hospital in test_loader:
            data, target, sensor, hospital = data.cuda(), target.long().cuda(), sensor.long().cuda(), \
                                             hospital.long().cuda()
            output, _ = model(data, sensor)
            if args.arch == 'CNNConStn':
                output_1, output_2 = torch.split(output, split_size_or_sections=output.shape[0] // 2)
                output = output_1
            if args.multiplier != 0:
                loss = sord_loss(logits=output, ground_truth=target, num_classes=nclasses, multiplier=args.multiplier)
            else:
                loss = criterion(output, target).mean()
            test_losses.append(loss.item())
            pred = F.softmax(output, dim=1).max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            # to compute metrics
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
    wandb.log(metrics, step=epoch)

    print('Saving weights...')
    save_weights(model, os.path.join(weights_path, 'model.pth'))
    save_best_model(model, weights_path, metrics, state_dict)

    # visualize the transformations
    if args.arch == 'CNNStn':
        stn_out = model.stn(fixed_samples.cuda()).cpu()
        viz_tensor = torch.cat([fixed_samples,
                                F.interpolate(stn_out, size=(args.img_size, args.img_size))], dim=3)
        save_image(viz_tensor, 'logs/viz_test/' + str(epoch).zfill(4) + '.png', nrow=int(viz_tensor.shape[0] ** 0.5))
    elif args.arch == 'CNNConStn':
        stn_out = F.interpolate(model.stn(fixed_samples.cuda())[0].cpu(), size=(args.img_size, args.img_size))
        stn_out_1, stn_out_2 = torch.split(stn_out, split_size_or_sections=stn_out.shape[0] // 2)
        viz_tensor = torch.cat([fixed_samples, stn_out_1, stn_out_2], dim=3)
        save_image(viz_tensor, os.path.join(args.test_viz_dir, str(epoch).zfill(4) + '.png'), nrow=int(viz_tensor.shape[0] ** 0.5))

def patient_inference(model, patient_loader, nclasses):
    model.eval()
    confusion_matrix = torch.zeros(nclasses, nclasses)
    with torch.no_grad():
        for data, target, _, _ in patient_loader:
            data = data.cuda()
            output = model(data)
            pred = F.softmax(output, dim=1).max(1, keepdim=True)[1]

            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    preds = confusion_matrix.diag()
    targets = confusion_matrix.sum(1)
    return preds, targets

def patient_histogram(model, patient_loader):
    model.eval()
    pred_prob = []
    pred_labels = []
    with torch.no_grad():
        for data, _ in patient_loader:
            data = data.cuda()
            output = model(data)
            pred = F.softmax(output, dim=1).max(1, keepdim=True)
            pred_prob.append(pred[0].view(-1).cpu())
            pred_labels.append(pred[1].view(-1).cpu())
    pred_prob = torch.cat(pred_prob).numpy()
    pred_labels = torch.cat(pred_labels).numpy()
    return pred_prob, pred_labels

def get_patient_dframe(pred_prob, pred_labels):
    df = pd.DataFrame({'pred_prob': pred_prob, 'pred_label': pred_labels})
    mapping = {0: 'no-covid', 1: 'covid'}
    df.replace({0: mapping, 1: mapping})
    return df

def compute_metrics(confusion_matrix):
    precision = confusion_matrix.diag()[0] / (confusion_matrix.sum(0)[0])
    recall = confusion_matrix.diag()[0] / (confusion_matrix.sum(1)[0])
    f1 = 2 * (recall * precision) / (recall + precision)
    return precision.item(), recall.item(), f1.item()

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

def check(args, point): # check the boundaries
    if point[0] <= 0:  # x
        point[0] = 0.00001  # for division
    if point[0] >= 1:
        point[0] = 1
    if point[1] <= 0:  # y
        point[1] = 0.00001
    if point[1] >= 1:
        point[1] = 1

    point[0] *= args.img_size
    point[1] *= args.img_size

    point[0] = point[0].astype(np.int)
    point[1] = point[1].astype(np.int)

    point = point.tolist()
    return point


def experiment(args):
    # load data
    data = pd.read_pickle(os.path.join(args.dataset_root, 'dataset.pkl'))
    #data = data[data.hospital.str.contains('|'.join(args.hospitals))]  # filter hospitals
    data = data[data.sensor.str.contains('|'.join(args.sensors))]  # filter sensors

    # load splits
    splits = pd.read_csv(os.path.join(args.dataset_root, 'splits', 'split_' + str(args.seed) + '.csv'))
    train_patients = splits[splits.split.str.contains('train')].patient_hash.tolist()
    test_patients = splits[splits.split.str.contains('test')].patient_hash.tolist()

    # get data accorting to patient split
    train_data = data[data.patient_hash.str.contains('|'.join(train_patients))]
    test_data = data[data.patient_hash.str.contains('|'.join(test_patients))]

    # subset the dataset
    train_dataset = COVID19Dataset(args, train_data, transforms_base=get_transforms(args, 'base_train'),
                                   transforms_to_tensor=get_transforms(args, 'to_tensor'),
                                   transforms_crop_1_train=get_transforms(args, 'crop_1_train'),
                                   transforms_crop_2_train=get_transforms(args, 'crop_2_train'))
    test_dataset = COVID19Dataset(args, test_data, transforms_base=get_transforms(args, 'base_test'),
                                  transforms_to_tensor=get_transforms(args, 'to_tensor'))

    # For unbalanced dataset we create a weighted sampler
    train_labels = [sum(l) for l in train_data.label.tolist()]
    weights = get_weights_for_balanced_classes(train_labels, len(list(set(train_labels))))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights, num_samples=len(weights))

    nclasses = len(list(set(train_labels)))
    nsensors = len(list(set(train_data.sensor.tolist())))
    nhospitals = len(list(set(train_data.hospital.tolist())))
    # dataloaders from subsets
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler if args.mode == 'train' else None,
        num_workers=args.num_workers,
        drop_last=True if args.mode == 'train' else False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=args.num_workers,
        drop_last=False)

    args.weights_dir = os.path.join('logs', args.wandb_name, 'weights')
    os.makedirs(args.weights_dir, exist_ok=True)

    args.train_viz_dir = os.path.join('logs', args.wandb_name, 'viz_train')
    args.test_viz_dir = os.path.join('logs', args.wandb_name, 'viz_test')
    os.makedirs(args.train_viz_dir, exist_ok=True)
    os.makedirs(args.test_viz_dir, exist_ok=True)

    if args.arch in ['CNN2D', 'CNNCon']:
        model = CNN2D(nclasses, use_stn=args.use_stn)
    elif args.arch == 'CNNStn':
        model = CNNStn(args.img_size, nclasses)
    elif args.arch == 'CNNConStn':
        model = CNNConStn(args.img_size, nclasses)
    elif args.arch == 'WideResnet':
        model = WideResnet(args.wrn_depth, args.wrn_width, args.dropout_rate, nclasses)
    elif args.arch == 'MobileNet':
        model = models.mobilenet_v2(pretrained=args.pretrained, num_classes=nclasses)
    elif args.arch == 'WhitenMobileNet':
        model = WhitenMobileNetV2(num_classes=nclasses, instance_whiten=args.whiten)
    elif args.arch == 'CNNProj':
        model = CNNProj(ndomains=nsensors)
    elif args.arch == 'ResNet50':
        model = models.resnet50(num_classes=nclasses, pretrained=False)
    else:
        raise NotImplementedError
    print(model)
    print('Number of params in the model: {}'.format(
        *[sum([p.data.nelement() for p in net.parameters()]) for net in [model]]))
    model = model.cuda()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.mode == 'train':
        wandb.init(project="covid", name=args.wandb_name, config=args)
        criterion = loss_criterion(nclasses=len(list(set(train_labels))), smoothing=args.smoothing)
        # fixed samples for visu
        fixed_samples_iter = iter(train_loader)
        fixed_samples_train, fixed_samples_train_dup, fixed_y_train, _, _ = fixed_samples_iter.next()
        fixed_samples = {}
        fixed_samples['train'] = fixed_samples_train
        fixed_samples['train_dup'] = fixed_samples_train_dup

        fixed_samples_iter = iter(test_loader)
        fixed_samples_test, _, _, _ = fixed_samples_iter.next()

        #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[70], gamma=0.1) # 10, 50
        state_dict = {'best_f1': 0., 'precision': 0., 'recall': 0., 'accuracy': 0.}
        for epoch in range(args.epochs):
            model = train(args, device, criterion, model, train_loader, len(list(set(train_labels))), optimizer, epoch, wandb,
                          fixed_samples, fixed_y_train)
            test(args, model, criterion, test_loader, len(list(set(train_labels))), epoch, state_dict, args.weights_dir, wandb,
                 fixed_samples_test)
            exp_lr_scheduler.step()
    elif args.mode == 'generate_csv':
        print('Loading weights from the checkpoint...{}'.format(args.weights_path))
        model = load_weights(args, model=model, path=args.weights_path).cuda()
        for split in ['train', 'test']:
            if split == 'train':
                df = train_data[['hospital', 'patient', 'raw_filenames', 'frame_pos', 'label']]
                train_labels = [sum(l) for l in train_data.label.tolist()]
                df['ground_truth'] = train_labels
                loader = train_loader
            elif split == 'test':
                df = test_data[['hospital', 'patient', 'raw_filenames', 'frame_pos', 'label']]
                test_labels = [sum(l) for l in test_data.label.tolist()]
                df['ground_truth'] = test_labels
                loader = test_loader

            model.eval()
            scores, preds = [], []
            with torch.no_grad():
                for data, _, target, sensor, hospital in loader:
                    data, target, sensor, hospital = data.cuda(), target.long().cuda(), sensor.long().cuda(), \
                                                     hospital.long().cuda()
                    output, _ = model(data, sensor)
                    if args.arch == 'CNNConStn':
                        output_1, output_2 = torch.split(output, split_size_or_sections=output.shape[0] // 2)
                        output = output_2
                    output_prob = F.softmax(output, dim=1)
                    scores.append(output_prob.cpu())
                    pred = output_prob.max(1, keepdim=True)[1]
                    preds.append(pred.view(-1).cpu())
            scores = torch.cat(scores, dim=0).numpy().tolist()
            preds = torch.cat(preds, dim=0).numpy().tolist()
            df['scores'] = scores
            df['prediction'] = preds
            df["filenames"] = df["raw_filenames"].map(lambda x: x[1])
            print('Saving the dataframe into a pkl and csv...')
            df.to_pickle(args.weights_path.split('/')[-3] + '_' + split + '.pkl')
            df.to_csv(args.weights_path.split('/')[-3] + '_' + split + '.csv')

    elif args.mode == 'generate_dataset':
        with torch.no_grad():
            counter = 0
            for data, target, sensor, hospital in test_loader:
                # take a batch of samples
                for label in range(nclasses):
                    # iterate through all the classes for their presence
                    label_idxs = (target == label)
                    if len(label_idxs) != 0: # if non zero
                        data_temp = data[label_idxs, ...] # extract the samples
                        for datum in data_temp: # iterate through the samples for that class
                            img = (np.transpose(datum.numpy(), (1, 2, 0)) * 255.).astype(np.uint8)
                            path = os.path.join('test_content', str(label), str(counter).zfill(5) + '.png')
                            print('Saving...{}'.format(path))
                            imageio.imsave(path, img)
                            counter += 1

    elif args.mode == 'bbox_video':
        print('Loading weights from the checkpoint...{}'.format(args.weights_path))
        model = load_weights(args, model=model, path=args.weights_path).cuda()
        model.eval()
        video_hashes = list(set(test_data.video_hash))
        video_hashes.sort()
        video_hash = video_hashes[args.constant]
        video_data = test_data[test_data.video_hash == video_hash]
        video_dataset = COVID19Dataset(args, video_data, get_transforms(args, 'test'))
        video_loader = torch.utils.data.DataLoader(video_dataset, batch_size=args.batch_size,
                                                   shuffle=False, sampler=None, num_workers=2,
                                                   drop_last=False)
        with torch.no_grad():
            out_frames = []
            for data, _, _, _ in video_loader:
                data = data.cuda()
                #data = affine_transform_images(args, data)
                output, _, thetas = model(data, None)
                print(thetas['theta_1'][0])
                grid_coord = F.affine_grid(thetas['theta_1'], data.shape)
                # normalize the co-ordinates
                for i in range(grid_coord.shape[0]):
                    grid_coord_theta = (grid_coord[i] + 1) / 2
                    org_img = (np.transpose(data[i].cpu().numpy(), (1, 2, 0)) * 255.).astype(np.uint8)

                    points_array = []
                    # for the first crop
                    grid_coord_theta = grid_coord_theta.cpu().numpy()
                    point1 = grid_coord_theta[0, 0, ...]
                    point1 = check(args, point1)
                    point2 = grid_coord_theta[0, args.img_size - 1, ...]
                    point2 = check(args, point2)
                    point3 = grid_coord_theta[args.img_size - 1, 0, ...]
                    point3 = check(args, point3)
                    point4 = grid_coord_theta[args.img_size - 1, args.img_size - 1, ...]
                    point4 = check(args, point4)

                    points_array.append(point1)
                    points_array.append(point2)
                    points_array.append(point4)
                    points_array.append(point3)

                    points_array = np.asarray(points_array, np.int32)
                    points_array = points_array.reshape((-1, 1, 2))

                    imageio.imsave('visu/temp.png', org_img)
                    img = cv2.imread('visu/temp.png')
                    cv2.polylines(img, [points_array], True, (255, 0, 0), thickness=2)

                    out_frames.append(torch.from_numpy(img).unsqueeze(0))
            out_frames = torch.cat(out_frames, dim=0)
        imageio.mimsave('output.mp4', out_frames, fps=5)


    elif args.mode == 'visualize':
        print('Loading weights from the checkpoint...{}'.format(args.weights_path))
        model = load_weights(args, model=model, path=args.weights_path).cuda()
        model.eval()

        points_array_1 = []
        points_array_2 = []
        fixed_points_array = np.asarray([[1, 1], [args.img_size - 1, 1],
                                         [args.img_size - 1, args.img_size - 1], [1, args.img_size - 1]], np.int32)
        fixed_points_array = fixed_points_array.reshape((-1, 1, 2))
        # load the image
        raw_image = img_as_float32(io.imread(args.img_path))
        data = torch.from_numpy(raw_image.transpose((2, 0, 1))).unsqueeze(0).cuda()
        with torch.no_grad():
            output, _, thetas = model(data, None)

            print(thetas['theta_1'][0].reshape(2, 3))
            print(thetas['theta_2'][0].reshape(2, 3))

            theta_1 = thetas['theta_1'][0].reshape(2, 3)
            theta_2 = thetas['theta_2'][0].reshape(2, 3)

            A_1, B_1 = theta_1[..., :2].cpu(), theta_1[..., -1].reshape(2, 1).cpu()
            A_2, B_2 = theta_2[..., :2].cpu(), theta_2[..., -1].reshape(2, 1).cpu()

            print(A_1, B_1)
            print(A_2, B_2)

            stn_out = F.interpolate(model.stn(data)[0].cpu(), size=(args.img_size, args.img_size))
            stn_out_1, stn_out_2 = torch.split(stn_out, split_size_or_sections=stn_out.shape[0] // 2)

            # save the cropped image
            path = os.path.join('visu', 'crop1.png')
            stn_out_1 = (np.transpose(stn_out_1[0].cpu().numpy(), (1, 2, 0)) * 255.).astype(np.uint8)
            imageio.imsave('visu/temp.png', stn_out_1) # save the temo image
            img = cv2.imread('visu/temp.png') # read the temp image
            cv2.polylines(img, [fixed_points_array], True, (255, 0, 0), thickness=3)
            imageio.imsave(path, img)

            # save the cropped image
            path = os.path.join('visu', 'crop2.png')
            stn_out_2 = (np.transpose(stn_out_2[0].cpu().numpy(), (1, 2, 0)) * 255.).astype(np.uint8)
            imageio.imsave('visu/temp.png', stn_out_2)  # save the temo image
            img = cv2.imread('visu/temp.png')  # read the temp image
            cv2.polylines(img, [fixed_points_array], True, (0, 255, 0), thickness=3)
            imageio.imsave(path, img)

            c1 = torch.FloatTensor([-1, -1]).reshape(2, 1)
            c2 = torch.FloatTensor([-1, 1]).reshape(2, 1)
            c3 = torch.FloatTensor([1, -1]).reshape(2, 1)
            c4 = torch.FloatTensor([1, 1]).reshape(2, 1)

            c1 = torch.floor(((torch.mm(A_1, c1) + B_1) + 1) * 224 / 2)
            c2 = torch.floor(((torch.mm(A_1, c2) + B_1) + 1) * 224 / 2)
            c3 = torch.floor(((torch.mm(A_1, c3) + B_1) + 1) * 224 / 2)
            c4 = torch.floor(((torch.mm(A_1, c4) + B_1) + 1) * 224 / 2)

            print('hahahaha')
            c1 = c1.clamp(1, 223)
            c2 = c2.clamp(1, 223)
            c3 = c3.clamp(1, 223)
            c4 = c4.clamp(1, 223)

            org_img = (np.transpose(data[0].cpu().numpy(), (1, 2, 0)) * 255.).astype(np.uint8)
            imageio.imsave('visu/temp.png', org_img)
            img = cv2.imread('visu/temp.png')
            cv2.drawMarker(img, (int(c1[0].numpy()), int(c1[1].numpy())), (0, 255, 0), markerType=cv2.MARKER_CROSS,
                                          markerSize=10, thickness=2)
            cv2.drawMarker(img, (int(c2[0].numpy()), int(c2[1].numpy())), (0, 255, 0), markerType=cv2.MARKER_CROSS,
                           markerSize=10, thickness=2)
            cv2.drawMarker(img, (int(c3[0].numpy()), int(c3[1].numpy())), (0, 255, 0), markerType=cv2.MARKER_CROSS,
                           markerSize=10, thickness=2)
            cv2.drawMarker(img, (int(c4[0].numpy()), int(c4[1].numpy())), (0, 255, 0), markerType=cv2.MARKER_CROSS,
                           markerSize=10, thickness=2)
            path = os.path.join('visu', 'market.png')
            imageio.imsave(path, img)

            grid_coord_theta_1 = F.affine_grid(thetas['theta_1'], data.shape)
            grid_coord_theta_2 = F.affine_grid(thetas['theta_2'], data.shape)

            #print(grid_coord_theta_1.min())
            #print(grid_coord_theta_1.max())

            # normalize the co-ordinates
            for i in range(grid_coord_theta_1.shape[0]):
                grid_coord_theta_1[i] = (grid_coord_theta_1[i] + 1) / 2

            #print(grid_coord_theta_1.min())
            #print(grid_coord_theta_1.max())

            # normalize the co-ordinates
            for i in range(grid_coord_theta_2.shape[0]):
                grid_coord_theta_2[i] = (grid_coord_theta_2[i] + 1) / 2

            # save the original image
            path = os.path.join('visu', 'original.png')
            org_img = (np.transpose(data[0].cpu().numpy(), (1, 2, 0)) * 255.).astype(np.uint8)
            imageio.imsave(path, org_img)

            # for the first crop
            grid_coord_theta_1 = grid_coord_theta_1.cpu().numpy()
            point1_1 = grid_coord_theta_1[0, 0, 0, ...]
            point1_1 = check(args, point1_1)
            point2_1 = grid_coord_theta_1[0, 0, args.img_size - 1, ...]
            point2_1 = check(args, point2_1)
            point3_1 = grid_coord_theta_1[0, args.img_size - 1, 0, ...]
            point3_1 = check(args, point3_1)
            point4_1 = grid_coord_theta_1[0, args.img_size - 1, args.img_size - 1, ...]
            point4_1 = check(args, point4_1)

            points_array_1.append(point1_1)
            points_array_1.append(point2_1)
            points_array_1.append(point4_1)
            points_array_1.append(point3_1)

            points_array_1 = np.asarray(points_array_1, np.int32)
            points_array_1 = points_array_1.reshape((-1, 1, 2))
            print(points_array_1)

            #img = np.transpose(img, (2, 0, 1))
            #img.astype(np.float32)
            imageio.imsave('visu/temp.png', org_img)
            img = cv2.imread('visu/temp.png')
            #cv2.drawMarker(img, (int(point4_1[0]), int(point4_1[1])), (0, 255, 0), markerType=cv2.MARKER_CROSS,
            #               markerSize=10, thickness=2)
            cv2.polylines(img, [points_array_1], True, (255, 0, 0), thickness=2)

            # for the second crop
            grid_coord_theta_2 = grid_coord_theta_2.cpu().numpy()
            point1_2 = grid_coord_theta_2[0, 0, 0, ...]
            point1_2 = check(args, point1_2)
            point2_2 = grid_coord_theta_2[0, 0, args.img_size - 1, ...]
            point2_2 = check(args, point2_2)
            point3_2 = grid_coord_theta_2[0, args.img_size - 1, 0, ...]
            point3_2 = check(args, point3_2)
            point4_2 = grid_coord_theta_2[0, args.img_size - 1, args.img_size - 1, ...]
            point4_2 = check(args, point4_2)

            points_array_2.append(point1_2)
            points_array_2.append(point2_2)
            points_array_2.append(point4_2)
            points_array_2.append(point3_2)

            points_array_2 = np.asarray(points_array_2, np.int32)
            points_array_2 = points_array_2.reshape((-1, 1, 2))
            print(points_array_2)

            # img = np.transpose(img, (2, 0, 1))
            # img.astype(np.float32)
            # cv2.drawMarker(img, (int(point4_1[0]), int(point4_1[1])), (0, 255, 0), markerType=cv2.MARKER_CROSS,
            #               markerSize=10, thickness=2)
            cv2.polylines(img, [points_array_2], True, (0, 255, 0), thickness=2)
            path = os.path.join('visu', 'boxes.png')
            imageio.imsave(path, img)



if __name__ == '__main__':
    cudnn.benchmark = True
    args = parse_arguments()
    print(args)
    experiment(args)
    
    

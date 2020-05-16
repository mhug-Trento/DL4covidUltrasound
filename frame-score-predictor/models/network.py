import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from torchvision.models.mobilenet import MobileNetV2

import numpy as np

from models.whitening import InstanceWhitening

class SimpleCNN(nn.Module):
    def __init__(self, use_stn=False, inp_channels=3, inp_size=(256, 256), out_dim=2):
        super(SimpleCNN, self).__init__()
        self.use_stn = use_stn
        self.inp_channels = inp_channels
        self.out_dim = out_dim
        self.h, self.w = inp_size
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=self.inp_channels, out_channels=32,
            kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3),
            stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3),
            stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
            stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),
            stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),
            stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),
            stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),
            stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3),
            stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3),
            stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3)
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3),
            stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3),
            stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.block7 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Linear(in_features=256, out_features=self.out_dim)
        if self.use_stn:
            # Spatial Transformer localization-network
            self.localization = nn.Sequential(
                nn.Conv2d(in_channels=self.inp_channels, out_channels=16,
                kernel_size=5),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )
            # calculate the spatial dimensions of the localization features
            # TODO: get rid of hard coding
            height, width = self.h, self.w
            for i in range(3):
                height = (height - 4) // 2
                width = (width - 4) // 2
            self.loc_height, self.loc_width = height, width
            # Regressor for the 3 x 2 affine matrix
            self.fc_loc = nn.Sequential(
                nn.Linear(in_features=32 * height * width, out_features=128),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=128, out_features=3*2)
            )
            # Initialize the weights/biases with identity transformations
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0],
            dtype=torch.float))
    
    # spatial transformer network forward
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * self.loc_height * self.loc_width)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x
    
    def forward(self, x):
        if self.use_stn:
            x = self.stn(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = F.avg_pool2d(x, kernel_size=x.shape[-2:])
        x = x.view(x.shape[0], -1)
        x = self.block7(x)
        x = F.dropout(x, training=self.training)
        x = self.out(x)
        return x

class ClsProjection(nn.Module):
    def __init__(self, dim_in, dim_out, ndomains):
        super(ClsProjection, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.ndomains = ndomains
        self.emb_weight = nn.Embedding(ndomains, dim_in * dim_out)
        self.emb_bias = nn.Embedding(ndomains, dim_out)
        self.agnostic_linear = nn.Linear(dim_in, dim_out)
        with torch.no_grad():
            self.emb_bias.weight.zero_()

    def forward(self, x, domain):
        weight = self.emb_weight(domain)
        bias = self.emb_bias(domain)
        bs, d_in = x.shape
        x_feat = x.view(bs, d_in, 1)
        out = torch.bmm(weight.view(bs, self.dim_out, self.dim_in), x_feat)
        out = out.view(bs, -1)
        out = out + bias.view(bs, -1)
        out = out + self.agnostic_linear(x)
        return out

class CNNProj(nn.Module):
    def __init__(self, ndomains):
        super(CNNProj, self).__init__()
        self.ndomains = ndomains
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=3),
            nn.BatchNorm2d(32),  # 48 corresponds to the number of input features it
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # IN remains unchanged during any pooling operation
            #nn.Dropout(p=0.3)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3)
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.AvgPool2d(kernel_size=4)  # paper: 8
        )
        self.block7 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.out_proj = ClsProjection(dim_in=256, dim_out=2, ndomains=ndomains)

    def forward(self, x, domains):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = F.avg_pool2d(x, kernel_size=x.shape[-2:])
        x = x.view(x.shape[0], -1)  # reshape the tensor
        x = F.dropout(self.block7(x), training=self.training)
        x = self.out_proj(x, domains)
        return x

class CNN2D(nn.Module):
    def __init__(self, nclasses, use_stn=False):
        super(CNN2D, self).__init__()
        self.use_stn = use_stn
        self.block1 = nn.Sequential(
            #nn.InstanceNorm2d(3),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=3),
            nn.BatchNorm2d(32),  # 48 corresponds to the number of input features it
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # IN remains unchanged during any pooling operation
            #nn.Dropout(p=0.3)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3)
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.AvgPool2d(kernel_size=4)  # paper: 8
        )

        self.block7 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Linear(256, nclasses)

    def forward(self, x, domain=None):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = F.avg_pool2d(x, kernel_size=x.shape[-2:])
        x = x.view(x.shape[0], -1)  # reshape the tensor
        x = F.dropout(self.block7(x), training=self.training)

        x = self.out(x)

        return x, 0

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                     bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                          padding=1, bias=True)
        )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out = out + self.shortcut(x)
        return out

class WideResnet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, nclasses):
        super(WideResnet, self).__init__()
        self.in_planes = 16
        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor
        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate,
                                       stride=1)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate,
                                       stride=2)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate,
                                       stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], nclasses)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks) - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.shape[-2])
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out

class WhitenMobileNetV2(MobileNetV2):
    def __init__(self, num_classes=1000, instance_whiten=False):
        super(WhitenMobileNetV2, self).__init__(num_classes)
        self.instance_whiten = instance_whiten
        if instance_whiten:
            self.whiten = InstanceWhitening()
    def forward(self, x):
        if self.instance_whiten:
            x = self.whiten(x)
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

class CNNStn(nn.Module):
    def __init__(self, img_size, nclasses):
        super(CNNStn, self).__init__()

        self.img_size = img_size
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),  # 48 corresponds to the number of input features it
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # IN remains unchanged during any pooling operation
            #nn.Dropout(p=0.3)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3)
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.AvgPool2d(kernel_size=4)  # paper: 8
        )

        self.block1_stn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),  # 48 corresponds to the number of input features it
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # IN remains unchanged during any pooling operation
            #nn.Dropout(p=0.3)
        )

        self.block2_stn = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block3_stn = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block4_stn = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block5_stn = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3)
        )

        self.block6_stn = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block7 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Linear(256, nclasses)

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([.5, 0., 0.2, 0., .5, 0.2], #[.5, 0., 0.2, 0., .5, 0.2]
                                                    dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.block1_stn(x)
        xs = self.block2_stn(xs)
        xs = self.block3_stn(xs)
        xs = self.block4_stn(xs)
        xs = self.block5_stn(xs)
        xs = self.block6_stn(xs)
        xs = xs.view(-1, 128 * 7 * 7)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        bs, c, _ , _ = x.size()
        h , w = self.img_size // 2, self.img_size // 2
        stn_out_size = (bs, c, h, w)
        grid = F.affine_grid(theta, stn_out_size)
        x = F.grid_sample(x, grid)
        thetas = {}
        thetas['theta_1'] = theta
        return x, thetas

    def forward(self, x, domains=None):
        x, thetas = self.stn(x)  # transform the input
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = F.avg_pool2d(x, x.shape[-2])
        x = x.view(x.shape[0], -1)  # reshape the tensor
        x = F.dropout(self.block7(x), training=self.training)

        x = self.out(x)

        return x, 0, thetas


class CNNConStn(nn.Module):
    def __init__(self, img_size, nclasses):
        super(CNNConStn, self).__init__()

        self.img_size = img_size
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),  # 48 corresponds to the number of input features it
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # IN remains unchanged during any pooling operation
            #nn.Dropout(p=0.3)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3)
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.AvgPool2d(kernel_size=4)  # paper: 8
        )

        self.block1_stn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),  # 48 corresponds to the number of input features it
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # IN remains unchanged during any pooling operation
            #nn.Dropout(p=0.3)
        )

        self.block2_stn = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block3_stn = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block4_stn = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.3)
        )

        self.block5_stn = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3)
        )

        self.block6_stn = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block7 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Linear(256, nclasses)

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 32),
            nn.ReLU(True)
        )
        self.trans = nn.Linear(32, 4)  # predict just translation params
        self.scaling = nn.Linear(32, 2)  # just the scaling parameter

        # Initialize the weights/bias with some priors
        self.trans.weight.data.zero_()
        self.trans.bias.data.copy_(torch.tensor([0.1, 0.3, 0.1, 0.3],  # [.5, 0., 0.2, 0., .5, 0.2]
                                                dtype=torch.float))

        self.scaling.weight.data.zero_()
        self.scaling.bias.data.copy_(torch.tensor([0.1, 0.1],  # [.5, 0., 0.2, 0., .5, 0.2]
                                                  dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        scaling = 0 # for just translation
        xs = self.block1_stn(x)
        xs = self.block2_stn(xs)
        xs = self.block3_stn(xs)
        xs = self.block4_stn(xs)
        xs = self.block5_stn(xs)
        xs = self.block6_stn(xs)
        xs = xs.view(-1, 128 * 7 * 7)
        xs = self.fc_loc(xs)
        # predict the params
        trans = self.trans(xs)
        scaling = F.sigmoid(self.scaling(xs))
        scaling_1, scaling_2 = torch.split(scaling, split_size_or_sections=scaling.shape[1] // 2, dim=1)
        bs = trans.shape[0]
        trans_1, trans_2 = torch.split(trans, split_size_or_sections=trans.shape[1] // 2, dim=1)
        # prepare theta for each resolution
        theta_1 = torch.cat([torch.eye(2, 2, device='cuda').view(1, 2, 2) * scaling_1.view(bs, 1, 1),
                             trans_1.view(bs, 2, 1)], dim=2)
        theta_2 = torch.cat([torch.eye(2, 2, device='cuda').view(1, 2, 2) * scaling_2.view(bs, 1, 1),
                             trans_1.view(bs, 2, 1)], dim=2)
        # get the shapes
        bs, c, _ , _ = x.size()
        h , w = self.img_size // 2, self.img_size // 2
        stn_out_size = (bs, c, h, w)
        # apply transformations
        grid_1 = F.affine_grid(theta_1, stn_out_size)
        grid_2 = F.affine_grid(theta_2, stn_out_size)
        x_1 = F.grid_sample(x, grid_1)
        x_2 = F.grid_sample(x, grid_2)
        x = torch.cat([x_1, x_2], dim=0)
        thetas = {}
        thetas['theta_1'] = theta_1
        thetas['theta_2'] = theta_2
        return x, scaling, thetas
        return x, scaling, thetas

    def forward(self, x, domains=None):
        x, scaling, thetas = self.stn(x)  # transform the input
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = F.avg_pool2d(x, x.shape[-2])
        x = x.view(x.shape[0], -1)  # reshape the tensor
        x = F.dropout(self.block7(x), training=self.training)
        x = self.out(x)
        return x, scaling, thetas

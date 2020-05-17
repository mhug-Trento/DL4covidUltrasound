import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNConStn(nn.Module):
    def __init__(self, img_size, nclasses, fixed_scale=True):
        super(CNNConStn, self).__init__()

        self.img_size = img_size
        self.fixed_scale = fixed_scale
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

        if fixed_scale: # scaling is kept fixed, only translation is learned
            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = nn.Sequential(
                nn.Linear(128 * 7 * 7, 32),
                nn.ReLU(True),
                nn.Linear(32, 4)  # predict just translation params
            )
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([0.3, 0.3, 0.2, 0.2], dtype=torch.float))
        else: # scaling, rotation and translation are learned
            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = nn.Sequential(
                nn.Linear(128 * 7 * 7, 32),
                nn.ReLU(True)
            )
            self.trans = nn.Linear(32, 4)  # predict translation params
            self.scaling = nn.Linear(32, 2)  # predict the scaling parameter
            self.rotation = nn.Linear(32, 4)  # predict the rotation parameters

            # Initialize the weights/bias with some priors
            self.trans.weight.data.zero_()
            self.trans.bias.data.copy_(torch.tensor([0.3, 0.3, 0.2, 0.2], dtype=torch.float))

            self.scaling.weight.data.zero_()
            self.scaling.bias.data.copy_(torch.tensor([0.5, 0.75], dtype=torch.float))

            self.rotation.weight.data.zero_()
            self.rotation.bias.data.normal_(0, 0.1)

    # Spatial transformer network forward function
    def stn(self, x):
        scaling = 0 # dummy variable for just translation
        xs = self.block1_stn(x)
        xs = self.block2_stn(xs)
        xs = self.block3_stn(xs)
        xs = self.block4_stn(xs)
        xs = self.block5_stn(xs)
        xs = self.block6_stn(xs)
        xs = xs.view(-1, 128 * 7 * 7)

        if self.fixed_scale:
            trans = self.fc_loc(xs)
            bs = trans.shape[0]
            trans_1, trans_2 = torch.split(trans, split_size_or_sections=trans.shape[1] // 2, dim=1)
            # prepare theta for each resolution
            theta_1 = torch.cat([(torch.eye(2, 2, device='cuda') * 0.5).view(1, 2, 2).repeat(bs, 1, 1),
                                 trans_1.view(bs, 2, 1)], dim=2)
            theta_2 = torch.cat([(torch.eye(2, 2, device='cuda') * 0.75).view(1, 2, 2).repeat(bs, 1, 1),
                                 trans_1.view(bs, 2, 1)], dim=2)
        else:
            xs = self.fc_loc(xs)
            # predict the scaling params
            scaling = F.sigmoid(self.scaling(xs))
            scaling_1, scaling_2 = torch.split(scaling, split_size_or_sections=scaling.shape[1] // 2, dim=1)
            # predict the translation params
            trans = self.trans(xs)
            bs = trans.shape[0]
            trans_1, trans_2 = torch.split(trans, split_size_or_sections=trans.shape[1] // 2, dim=1)
            # predict the rotation params
            rot = self.rotation(xs)
            rot_1, rot_2 = torch.split(rot, split_size_or_sections=rot.shape[1] // 2, dim=1)
            # prepare theta for each resolution
            rot_1 = torch.ones(2, 2, device='cuda').fill_diagonal_(0).view(1, 2, 2).repeat(bs, 1, 1) * rot_1.view(bs, 2,
                                                                                                                  1)
            rot_2 = torch.ones(2, 2, device='cuda').fill_diagonal_(0).view(1, 2, 2).repeat(bs, 1, 1) * rot_2.view(bs, 2,
                                                                                                                  1)
            # add to the scaling params
            rot_1 = rot_1 + torch.eye(2, 2, device='cuda').view(1, 2, 2) * scaling_1.view(bs, 1, 1)
            rot_2 = rot_2 + torch.eye(2, 2, device='cuda').view(1, 2, 2) * scaling_2.view(bs, 1, 1)
            # prepare the final theta
            theta_1 = torch.cat([rot_1, trans_1.view(bs, 2, 1)], dim=2)
            theta_2 = torch.cat([rot_2, trans_1.view(bs, 2, 1)], dim=2)

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

        return x, scaling

    def forward(self, x, domains=None):
        x, scaling = self.stn(x)  # transform the input
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
        return x, scaling

from torch.utils.data import Dataset
import torch

import os
import pandas as pd
import numpy as np


class COVID19Dataset(Dataset):

    def __init__(self, args, data, transforms=None):
        self.dataset_root = args.dataset_root
        self.transforms = transforms
        self.data = data

    def __len__(self):
        return len(self.data.index)

    def __repr__(self):
        return repr(self.data)

    def __getitem__(self, idx):
        frame_file = self.data.iloc[idx].filename
        frame_path = os.path.join(self.dataset_root, 'frames', frame_file)
        frame = np.load(frame_path)
        if self.transforms:
            frame = self.transforms(frame)
        label = torch.tensor(sum(self.data.iloc[idx].label), dtype=torch.long)
        return frame, label
from torch.utils.data import Dataset
import torch

import os
import pandas as pd
import numpy as np


class COVID19Dataset(Dataset):

    def __init__(self, args, data, transforms=None, transforms_duplicate=None):
        self.dataset_root = args.dataset_root
        self.transforms = transforms
        self.data = data
        self.sensors = {'convex': 0, 'linear': 1, 'unknown': 2}
        self.hospitals = {'Lucca': 0,
                          'Brescia': 1,
                          'Pavia': 2,
                          'No Covid Data': 3,
                          'Germania': 4,
                          'Gemelli - Roma': 5,
                          'Trento': 6,
                          'Tione': 7}

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
        #label = torch.tensor(self.data.iloc[idx].label[0], dtype=torch.long)
        label = torch.tensor(sum(self.data.iloc[idx].label), dtype=torch.long)
        sensor_type = self.data.iloc[idx].sensor
        sensor = torch.tensor(self.sensors[sensor_type], dtype=torch.long)
        hospital_name = self.data.iloc[idx].hospital
        hospital = torch.tensor(self.hospitals[hospital_name], dtype=torch.long)
        return frame, label, sensor, hospital
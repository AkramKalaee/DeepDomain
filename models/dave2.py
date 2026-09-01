#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:52:33 2021

@author: chingis
"""

import sys

import torch
import torch.nn as nn
from PIL import Image
from itertools import product
from easydict import EasyDict as edict
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from AI_self_driving_car_main.DataLoading import EvalDataset as ED
from AI_self_driving_car_main.DataLoading import ConsecutiveBatchSampler as CB
import json
import numpy as np
from AI_self_driving_car_main.model.SimpleTransformer import SimpleTransformer

class dave2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers=nn.Sequential(
            nn.Conv2d(3,24,5,stride=2),
            nn.ELU(),
            nn.Conv2d(24,36,5,stride=2),
            nn.ELU(),
            nn.Conv2d(36,48,5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.5)   
        )
        self.dense_layers=nn.Sequential(
            nn.Linear(in_features=64 * 8 * 33, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1)
        )
    def forward(self,data):
        # print(data.shape)
        output = self.conv_layers(data)
        output = output.view(output.size(0), -1)
        output = self.dense_layers(output)
        return output

    def predict_by_image_path(self, root_dir, csv_file, parameters, device):
        validation_set = ED.EvalDataset(csv_file=csv_file,
                                        root_dir=root_dir,
                                        transform=transforms.Compose([
                                            # transforms.Resize(parameters.image_size),
                                            transforms.ToTensor(),
                                            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            #                      std=[0.229, 0.224, 0.225])
                                            transforms.Normalize(*parameters.normalization)
                                            # transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            #                      std=[0.5, 0.5, 0.5])
                                            # transforms.Lambda(lambda x: (x / 255.0) - 0.5)
                                        ]),
                                        public=False,
                                        private=False,
                                        img_size=parameters.image_size,
                                        optical_flow=False)

        validation_loader = DataLoader(validation_set, shuffle=False, num_workers=parameters.num_workers,
                                       batch_size=parameters.batch_size)

        # self.eval()
        # Calculation on Validation Loss
        with torch.no_grad():
            for Validation_sample in validation_loader:
                param_values = [v for v in Validation_sample.values()]
                image, angle, idx = param_values
                image = image.to(device)
                prediction = self(image)
                # result = prediction.reshape(-1, 1).detach().cpu().flatten().numpy().tolist()[0]

        # result = torch.tensor([result])
        # return result, image
        return prediction

    def get_tensor(self, root_dir, csv_file, parameters, device):
        validation_set = ED.EvalDataset(csv_file=csv_file,
                                        root_dir=root_dir,
                                        transform=transforms.Compose([
                                            # transforms.Resize(parameters.image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(*parameters.normalization)
                                        ]),
                                        optical_flow=False,
                                        img_size=parameters.image_size)

        validation_loader = DataLoader(validation_set, shuffle=False, num_workers=parameters.num_workers,
                                       batch_size=1)

        for Validation_sample in validation_loader:
            param_values = [v for v in Validation_sample.values()]
            image, angle, idx = param_values
            image = image.to(device)
        return image

    # def predict_by_image_data(self, input, device):
    #     image = input.to(device)
    #     angle_hat = self(image)
    #     # result = angle_hat.reshape(-1, 1).detach().cpu().flatten().numpy().tolist()[0]
    #     #
    #     # result = torch.tensor([result])
    #     return angle_hat

if __name__ == '__main__':
    car = DAVE2()
    x = torch.randn(32, 3, 120, 320)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:44:16 2021

@author: chingis
"""

import torch
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
from itertools import product
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tools.DEEPDOMAIN.AI_self_driving_car_main.DataLoading import EvalDataset as ED
from tools.DEEPDOMAIN.AI_self_driving_car_main.DataLoading import ConsecutiveBatchSampler as CB
from tools.DEEPDOMAIN.AI_self_driving_car_main.model.SimpleTransformer import SimpleTransformer


class SequenceModel(nn.Module):
    # def __init__(self, seq_len):
    #     self.seq_len = seq_len
    #     super(SequenceModel,self).__init__()
    #     self.ResNet = models.resnet50(pretrained=True)
    #
    #     self.fc0 = nn.Linear(in_features =1000, out_features = 512, bias=True)
    #     self.relu1 = nn.ReLU()
    #
    #     self.LSTM1 = nn.LSTM(input_size = 512, hidden_size = 256, num_layers = 2, batch_first=True)
    #     #self.LSTM2 = nn.LSTM(input_size = 256, hidden_size = 256, num_layers = 1, batch_first=True)
    #
    #     self.fc1 = nn.Linear(in_features =256, out_features = 64, bias=True)
    #     self.relu2 = nn.ReLU()
    #
    #     #self.fc2 = nn.Linear(in_features = 128, out_features = 128, bias=True)
    #     #self.fc3 = nn.Linear(in_features = 128, out_features = 64, bias=True)
    #     self.fc4 = nn.Linear(in_features = 64, out_features = 16, bias=True)
    #     self.relu3 = nn.ReLU()
    #
    #     self.fc5 = nn.Linear(in_features = 16, out_features = 1, bias=True)
    #     self.dropout = nn.Dropout(0.5)

    def __init__(self, seq_len):
        self.seq_len = seq_len
        super(SequenceModel, self).__init__()
        self.ResNet = models.resnet50(pretrained=True)

        self.fc0 = nn.Linear(in_features=1000, out_features=512, bias=True)

        self.LSTM1 = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)
        # self.LSTM2 = nn.LSTM(input_size = 256, hidden_size = 256, num_layers = 1, batch_first=True)

        self.fc1 = nn.Linear(in_features=256, out_features=64, bias=True)

        # self.fc2 = nn.Linear(in_features = 128, out_features = 128, bias=True)
        # self.fc3 = nn.Linear(in_features = 128, out_features = 64, bias=True)
        self.fc4 = nn.Linear(in_features=64, out_features=16, bias=True)

        self.fc5 = nn.Linear(in_features=16, out_features=1, bias=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, Input):
        x = Input.reshape(-1, 3, 224, 224)
       # print(x.shape)
        x = F.relu(self.fc0(self.ResNet(x)))
        image = x.reshape(-1,  self.seq_len, 512)
       # print(image.shape)
        h = torch.zeros(2, image.shape[0], 256).cuda()
        c = torch.zeros(2, image.shape[0], 256).cuda()
        image = self.LSTM1(image, (h,c))
        image = image[0]
        image = torch.tanh(image)
        
        # LSTM 16
        #image = torch.tanh(self.LSTM2(image)[0])

        # FC 512
        image = image.reshape(-1, 256)
        image = F.relu(self.fc1(image))
 
        # FC 128
        #image = F.relu(self.fc2(image))

        # FC 64
        #image = F.relu(self.fc3(image))
 
        # FC 16
        image = F.relu(self.fc4(image))

        # FC 1
        angle = self.fc5(image)

        return angle

    def forward_v1(self, Input):
        x = Input.reshape(-1, 3, 224, 224)
       # print(x.shape)
        x = self.relu1(self.fc0(self.ResNet(x)))
        image = x.reshape(-1,  self.seq_len, 512)
       # print(image.shape)
        h = torch.zeros(2, image.shape[0], 256).cuda()
        c = torch.zeros(2, image.shape[0], 256).cuda()
        image = self.LSTM1(image, (h,c))
        image = image[0]
        image = torch.tanh(image)

        # LSTM 16
        # image = torch.tanh(self.LSTM2(image)[0])

        # FC 512
        image = image.reshape(-1, 256)
        image = self.relu2(self.fc1(image))

        # FC 128
        #image = F.relu(self.fc2(image))

        # FC 64
        #image = F.relu(self.fc3(image))

        # FC 16
        image = self.relu3(self.fc4(image))

        # FC 1
        angle = self.fc5(image)

        return angle



    def predict_by_image_path(self, root_dir, csv_file, parameters, device):
        validation_set = ED.EvalDataset(csv_file=csv_file,
                                        root_dir=root_dir,
                                        transform=transforms.Compose([
                                            # transforms.Resize(tuple(parameters.image_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(*parameters.normalization)
                                        ]),
                                        img_size=parameters.image_size,
                                        optical_flow=parameters.optical_flow,
                                        )

        validation_cbs = CB.ConsecutiveBatchSampler(data_source=validation_set, batch_size=parameters.batch_size,
                                                    shuffle=False, drop_last=False, seq_len=parameters.seq_len,
                                                    use_all_frames=parameters.all_frames)
        validation_loader = DataLoader(validation_set, sampler=validation_cbs, num_workers=parameters.num_workers,
                                       collate_fn=(lambda x: x[0]))

        self.eval()
        with torch.no_grad():
            for Validation_sample in validation_loader:
                param_values = [v for v in Validation_sample.values()]
                if parameters.optical_flow:
                    image, angle, idx, optical = param_values
                    optical = optical.to(device)
                else:
                    image, angle, idx = param_values
                cur_bn = image.shape[0]
                # image = image.permute(0,2,1,3,4)
                # optical = optical.permute(0,2,1,3,4)
                loss = 0
                image = image.to(device)
                # optical = optical.to(device)
                if parameters.optical_flow:
                    angle_hat, _ = self(image, optical)

                else:
                    angle_hat = self(image)
                angle_hat = angle_hat.reshape(-1, 1)

                predictions = angle_hat.detach().cpu().flatten().numpy().tolist()

        result = predictions[0]
        for prediction in predictions[1:]:
            result = result * 0.5 + 0.5 * prediction

        image = image[0][0].unsqueeze(0)
        result = torch.tensor([result])
        return result, image

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

        self.eval()
        with torch.no_grad():
            for Validation_sample in validation_loader:
                param_values = [v for v in Validation_sample.values()]
                if parameters.optical_flow:
                    image, angle, idx, optical = param_values
                    optical = optical.to(device)
                else:
                    image, angle, idx = param_values
                image = image.to(device)
        return image

    def predict_by_image_data(self, input, device):
        def prepare_intput(input):
            image = torch.cat(tuple([input for i in range(self.seq_len)]), dim=0).unsqueeze(0)
            return image
        # image = prepare_intput(input)
        self.seq_len = 1
        image = input.to(device)
        result = self(image)
        # predictions = angle_hat.detach().cpu().flatten().numpy().tolist()

        # result = predictions[0]
        # for prediction in predictions[1:]:
        #     result = result * 0.5 + 0.5 * prediction
        # result = torch.tensor([result])
        return result
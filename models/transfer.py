#!/usr/bin/env python
# coding: utf-8

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from AI_self_driving_car_main.DataLoading import EvalDataset as ED
from AI_self_driving_car_main.DataLoading import ConsecutiveBatchSampler as CB

class transfer(nn.Module):
    def __init__(self):
        super(transfer,self).__init__()
        self.ResNet = models.resnet50(pretrained=True, progress=True)
        
        # self.fc1 = nn.Linear(in_features =1000, out_features = 512, bias=True)
        # self.fc2 = nn.Linear(in_features = 512, out_features = 256, bias=True)
        # self.fc3 = nn.Linear(in_features = 256, out_features = 64, bias=True)
        # self.fc4 = nn.Linear(in_features = 64, out_features = 1, bias=True)

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=1000, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1, bias=True)
        )

        # for name, param in self.ResNet.named_parameters():
        #     if 'layer' in name:
        #         param.requires_grad = False

    def forward(self, Input):
        # image = self.ResNet(Input)
        # # input size = (1,3,224,224)-(Batches, Channels, Height, Width)
        # # output size = (1, 1000)-(Batches, Feature)
        # image = F.relu(self.fc1(image))
        # # input size = (1,1000)-(batches, features)
        # # output size = (1,512)-(batches, features)
        # image = F.relu(self.fc2(image))
        # # input size = (1,512)-(batches, features)
        # # output size = (1,256)-(batches, features)
        # image = F.relu(self.fc3(image))
        # # input size = (1,256)-(batches, features)
        # # output size = (1,64)-(batches, features)
        # angle = self.fc4(image)
        # # input size = (1,64)-(batches, features
        # # output size = (1,1)-(batches, features)
        # return angle

        image = self.ResNet(Input)
        angle = self.fc_layers(image)
        return angle

    def predict_by_image_path(self, root_dir, csv_file, parameters, device):
        validation_set = ED.EvalDataset(csv_file=csv_file,
                                        root_dir=root_dir,
                                        transform=transforms.Compose([
                                            # transforms.Resize(parameters.image_size),
                                            transforms.ToTensor(),
                                            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            #                      std=[0.229, 0.224, 0.225])
                                            transforms.Normalize(0.5, 0.5),
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
        with torch.no_grad():
            for Validation_sample in validation_loader:
                param_values = [v for v in Validation_sample.values()]
                image, angle, idx = param_values
                image = image.to(device)
                prediction = self(image)
                # result = prediction.reshape(-1, 1).detach().cpu().flatten().numpy().tolist()[0]

        # result = torch.tensor([result])
        # return prediction, image
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
    #
    #     # result = torch.tensor([result])
    #     return angle_hat


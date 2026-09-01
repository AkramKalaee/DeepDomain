#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 12:54:17 2021

@author: chingis
"""
import sys
from easydict import EasyDict as edict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from AI_self_driving_car_main.model.TransferLearning import TLearning
import torch.optim as optim
from AI_self_driving_car_main.DataLoading import UdacityDataset as UD
from AI_self_driving_car_main.model.DAVE2 import DAVE2
from torchvision import transforms
# import wandb
import os
from torch.utils.tensorboard import SummaryWriter

from constants import SRC_DIR


# os.environ["WANDB_SILENT"] = "true"
# os.environ["HTTPS_PROXY"] = "proxy_url"
# noinspection PyAttributeOutsideInit


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

device = torch.device("cuda")

parameters = edict(
    learning_rate = 0.001,
    batch_size = 32,
    seq_len = 5,
    num_workers = 0, #4,
    model_name = 'transfer',
    image_size = (120,320)
)
if not os.path.exists( f'saved_models/{parameters.model_name}'):
    os.makedirs(f'saved_models/{parameters.model_name}')
if parameters.model_name == 'dave2':
    model = DAVE2()
elif parameters.model_name == 'transfer':
    model = TLearning()
else:
    raise KeyError("Unknown Architecture")

network = model.to(device)

writer = SummaryWriter()
# wandb.init(config=parameters, project='self-driving-car')
# wandb.watch(network)
optimizer = optim.Adam(network.parameters(),lr = parameters.learning_rate,betas=(0.9, 0.999), eps=1e-08)

udacity_dataset = UD.UdacityDataset(csv_file=f'{SRC_DIR}/dataset/train/labels.csv',
                             root_dir=f'{SRC_DIR}/dataset/train/',
                            transform=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()]),
                            select_camera='center')

dataset_size = int(len(udacity_dataset))
del udacity_dataset
split_point = int(dataset_size * 0.9)

training_set = UD.UdacityDataset(csv_file=f'{SRC_DIR}/dataset/train/labels.csv',
                             root_dir=f'{SRC_DIR}/dataset/train/',
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)
                                ]),
                            img_size=parameters.image_size,
                            optical_flow=False,
                            select_camera='center',
                            select_range=(0,split_point))
validation_set = UD.UdacityDataset(csv_file=f'{SRC_DIR}/dataset/train/labels.csv',
                             root_dir=f'{SRC_DIR}/dataset/train/',
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)]),
                            img_size=parameters.image_size,
                            optical_flow=False,
                            select_camera='center',
                            select_range=(split_point,dataset_size))

training_loader = DataLoader(training_set, shuffle=True, num_workers=parameters.num_workers, batch_size=parameters.batch_size)
validation_loader = DataLoader(validation_set,  shuffle=False, num_workers=parameters.num_workers, batch_size=parameters.batch_size)
criterion =  torch.nn.MSELoss()
criterion.to(device)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = parameters.learning_rate
    if epoch in [30, 90, 150]:
        lr = parameters.learning_rate * 0.1
        parameters.learning_rate = lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(160):
    losses = AverageMeter()
    torch.save(network.state_dict(), f'saved_models/{parameters.model_name}/epoch_{epoch}.tar')
    network.train()
    adjust_learning_rate(optimizer, epoch)
    # Calculation on Training Loss
    for training_sample in tqdm(training_loader):
        param_values = [v for v in training_sample.values()]
        image, angle = param_values
        cur_bn = image.shape[0]
        image = image.to(device)

        prediction = network(image)
        prediction = prediction.reshape(-1, 1)
        labels = angle.float().reshape(-1, 1).to(device)
        training_loss_angle = criterion(prediction,labels)

        losses.update(training_loss_angle.item())
        optimizer.zero_grad()
        training_loss_angle.backward()
        optimizer.step()
    
    print("Done")
    network.eval()
    # Calculation on Validation Loss
    val_losses = AverageMeter()
    with torch.no_grad():    
        for Validation_sample in tqdm(validation_loader):
            param_values = [v for v in Validation_sample.values()]
            image, angle = param_values
            cur_bn = image.shape[0]
            image = image.to(device)

            prediction = network(image)
            prediction = prediction.reshape(-1,1)
            labels = angle.float().reshape(-1,1).to(device)

            validation_loss_angle = torch.sqrt(criterion(prediction,labels)+ 1e-6)
            val_losses.update(validation_loss_angle.item())
    # wandb.log({'training_loss': losses.avg, 'val_loss': val_losses.avg})

    # Log scalar values
    writer.add_scalar('training_loss', losses.avg, global_step=epoch)
    writer.add_scalar('val_loss', val_losses.avg, global_step=epoch)

    # Log histograms
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, global_step=epoch)
    print(f"training_loss: {losses.avg} val_loss: {val_losses.avg}")

        
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 13:05:49 2021

@author: chingis
"""
from PIL import Image
from itertools import product
from easydict import EasyDict as edict
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from constants import SRC_DIR
from tools.DEEPDOMAIN.AI_self_driving_car_main.model.MotionTransformer import MotionTransformer
from tools.DEEPDOMAIN.AI_self_driving_car_main.model.LSTM import SequenceModel
from tools.DEEPDOMAIN.AI_self_driving_car_main.DataLoading import EvalDataset as ED
from tools.DEEPDOMAIN.AI_self_driving_car_main.DataLoading import ConsecutiveBatchSampler as CB
import json
import numpy as np
from tools.DEEPDOMAIN.AI_self_driving_car_main.model.SimpleTransformer import SimpleTransformer
#import wandb
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



if __name__ == "__main__":
    device = torch.device("cuda")

    parameters = edict(
        batch_size = 13,
        seq_len = 5,
        num_workers = 16,
        model_name = 'MotionTransformer',
        normalization = ([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        image_size=(224, 224),
        all_frames=True,
        optical_flow=True,
        checkpoint=f'{SRC_DIR}/tools/DEEPDOMAIN/AI_self_driving_car_main/saved_models/transformer/opticaltransformer.tar'
    )


    if parameters.model_name == 'LSTM':
        model_object = SequenceModel
    elif parameters.model_name == 'MotionTransformer' :
        model_object = MotionTransformer
    elif parameters.model_name == 'SimpleTransformer' :
        model_object = SimpleTransformer
    else:
        raise KeyError("Unknown Architecture")

    network = model_object(parameters.seq_len)
    network.load_state_dict(torch.load(parameters.checkpoint))
    network.to(device)
    #wandb.init(config=parameters, project='self-driving-car')
    #wandb.watch(network)

    validation_set = ED.EvalDataset(csv_file=f'{SRC_DIR}/dataset/test/labels.csv',
                                     root_dir=f'{SRC_DIR}/dataset/test/center/',
                                 transform=transforms.Compose([
                                    # transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(*parameters.normalization)
                                     ]),
                                 img_size=parameters.image_size,
                                 optical_flow=parameters.optical_flow,
                                 )



    validation_cbs = CB.ConsecutiveBatchSampler(data_source=validation_set, batch_size=parameters.batch_size, shuffle=False, drop_last=False, seq_len=parameters.seq_len, use_all_frames=parameters.all_frames)
    validation_loader = DataLoader(validation_set, sampler=validation_cbs, num_workers=parameters.num_workers, collate_fn=(lambda x: x[0]))
    criterion =  torch.nn.MSELoss()
    criterion.to(device)

    predictions = []
    network.eval()
    val_angle_losses = AverageMeter()
    with torch.no_grad():
        for Validation_sample in tqdm(validation_loader):
            param_values = [v for v in Validation_sample.values()]
            if parameters.optical_flow:
                image, angle, idx,  optical = param_values
                optical = optical.to(device)
            else:
                image, angle, idx = param_values
            cur_bn = image.shape[0]
            #image = image.permute(0,2,1,3,4)
            #optical = optical.permute(0,2,1,3,4)
            loss = 0
            image = image.to(device)
            #optical = optical.to(device)
            if parameters.optical_flow:
                angle_hat, _ = network(image, optical)

            else:
                angle_hat = network(image)
            angle_hat = angle_hat.reshape(-1, 1)
            angle = angle.float().reshape(-1, 1).to(device)

            validation_loss_angle = torch.sqrt(criterion(angle_hat,angle) + 1e-6)
            loss += validation_loss_angle

            val_angle_losses.update(validation_loss_angle.item())
            predictions.extend(list(zip(idx.detach().cpu().flatten().numpy().tolist(), angle_hat.detach().cpu().flatten().numpy().tolist() )))


    print(val_angle_losses.avg)

    res_dic = {}
    for pair in predictions:
        idx, angle = pair
        if idx not in res_dic.keys():
            res_dic[idx] = angle
        else:
            res_dic[idx] = res_dic[idx] * 0.5 + 0.5 * angle


    a_file = open("predictions.json", "w")
    a_file = json.dump(res_dic, a_file)

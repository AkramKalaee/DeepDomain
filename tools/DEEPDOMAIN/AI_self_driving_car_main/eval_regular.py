#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 13:05:49 2021

@author: chingis
"""
from constants import SRC_DIR

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 12:24:40 2021

@author: chingis
"""
import cv2
import copy
from PIL import Image
from itertools import product
from easydict import EasyDict as edict
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tools.DEEPDOMAIN.AI_self_driving_car_main.model.TransferLearning import TLearning
from tools.DEEPDOMAIN.AI_self_driving_car_main.model.DAVE2 import DAVE2
from tools.DEEPDOMAIN.AI_self_driving_car_main.DataLoading import EvalDataset as ED
from tools.DEEPDOMAIN.AI_self_driving_car_main.DataLoading import ConsecutiveBatchSampler as CB
import json
import numpy as np

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
        batch_size = 64,
        num_workers = 8,
        image_size = (120,320),
        model_name = 'dave2',
        checkpoint=f'{SRC_DIR}/tools/DEEPDOMAIN/AI_self_driving_car_main/saved_models/DAVE2/DAVE2.tar'
    )


    if parameters.model_name == 'dave2':
        model_object = DAVE2
    elif parameters.model_name == 'transfer':
        model_object = TLearning
    else:
        raise KeyError("Unknown Architecture")

    network = model_object()
    network.to(device)
    network.load_state_dict(torch.load(parameters.checkpoint))
    network.eval()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(parameters.image_size),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: (x / 255.0) - 0.5),

        transforms.Normalize(0.5, 0.5),
        # transforms.Lambda(lambda x: (x / 127.5) - 1.0),
        #

        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
        # transforms.Lambda(lambda x: (x / 127.5) - 1.0)
        #                                   ]
        # transforms.Normalize(
        #     mean=[0.45734706, 0.43338275, 0.40058118],
        #     std=[0.23965294, 0.23532275, 0.2398498],
        #     )
    ])
    image_path = f'{SRC_DIR}/dataset/test/center/1479425542450284490.jpg'

    frame = cv2.imread(image_path)
    # orig_frame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame[65:-25, :, :]
    frame = transform(frame)
    frame = frame.unsqueeze(0).to(device)
    outputs = network(frame)
    print(frame.shape)
    print(np.float64(outputs.item()))

    # image_data = cv2.resize(cv2.imread(image_path), tuple(parameters.image_size))
    # current_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    # image_data = current_image[65:-25, :, :]

    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = image[65:-25, :, :]
    # image = cv2.resize(image, tuple(parameters.image_size))
    # # pixels = np.asarray(image)
    # # convert from integers to floats
    # # pixels = pixels.astype('float32')
    # # # calculate global mean and standard deviation
    # # mean, std = pixels.mean(), pixels.std()
    # # print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    # # # global standardization of pixels
    # # pixels = (pixels - mean) / std
    # # # clip pixel values to [-1,1]
    # # # pixels = np.clip(pixels, -1.0, 1.0)
    # # # # shift from [-1,1] to [0,1] with 0.5 mean
    # # # pixels = (pixels + 0.5) / 2.0
    #
    # img = transform(image)
    #
    # #
    # # img = torch.from_numpy(copy.deepcopy(image_data))
    # # img = img.to(torch.float)
    # # img = img.permute(2, 0, 1)
    # # transform = transforms.Compose([
    # #     # transforms.Resize((224,224)),
    # #     # transforms.ToTensor(),
    # #     transforms.Lambda(lambda x: (x / 255.0) - 0.5),
    # #     # transforms.Normalize(0.5, 0.5),
    # #     # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    # #     #                      std=[0.229, 0.224, 0.225])
    # # ])
    # # tensor_data = transform(img)
    # image = img.unsqueeze(0).cuda().detach()
    # img = image.to(device)
    # network.eval()
    # outputs = network(img)
    # print(outputs.item())
    # exit(0)

    validation_set = ED.EvalDataset(csv_file=f'{SRC_DIR}/dataset/test/labels.csv',
                                 root_dir=f'{SRC_DIR}/dataset/test/center/',
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

    validation_loader = DataLoader(validation_set,  shuffle=False, num_workers=parameters.num_workers, batch_size=parameters.batch_size)
    criterion = torch.nn.MSELoss()
    criterion.to(device)

    network.eval()
    predictions = []
    # Calculation on Validation Loss
    val_losses = AverageMeter()
    with torch.no_grad():
        for Validation_sample in tqdm(validation_loader):
            param_values = [v for v in Validation_sample.values()]
            image, angle, idx = param_values
            cur_bn = image.shape[0]
            image = image.to(device)
            prediction = network(image)
            prediction = prediction.reshape(-1,1)
            labels = angle.float().reshape(-1,1).to(device)


            validation_loss_angle = torch.sqrt(criterion(prediction,labels)+ 1e-6)
            val_losses.update(validation_loss_angle.item())

            predictions.extend(list(zip(idx.detach().cpu().flatten().numpy().tolist(), prediction.detach().cpu().flatten().numpy().tolist() )))


    print(val_losses.avg)
    res_dic = {}
    for pair in predictions:
        idx, angle = pair
        if idx not in res_dic.keys():
            res_dic[idx] = angle
            print(idx, angle)

    a_file = open("predictions.json", "w")
    a_file = json.dump(res_dic, a_file)
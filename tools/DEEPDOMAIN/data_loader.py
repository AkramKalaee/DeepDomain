import csv
import glob
import os
import random

import cv2
import numpy as np
from PIL import Image
import json
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
# from torchtext import data
# from torchtext import datasets

import constants


class UDACITYDataset(Dataset):
    def __init__(self,
                 args,
                 image_dir=constants.UDACITY_DATASET_DIR,
                 label2index_file=constants.UDACITY_DATASET_LABEL_TO_INDEX,
                 split='train'):
        super(UDACITYDataset).__init__()
        assert split in ['train', 'test', 'seed']
        # self.total_class_num = 10
        self.args = args


        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
            transforms.Resize(self.args.image_size),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
        ])

        self.image_list = []

        # if split == "train":
        #     self.class_list = ['left', 'center', 'right'] #sorted(os.listdir(self.image_dir))[:self.args.num_class]
        # else:
        #     self.class_list = ["center"]


        if not os.path.exists(self.image_dir):
            self.class_list = []
            self.label2index_file = image_dir + label2index_file
            sub_dir = self.image_dir.strip('/')
            self.image_dir = ""
            self.image_dir_list = []
            unique_images_dir = sub_dir[0:-5]
            unique_images_list = pd.read_csv(f"{unique_images_dir}/unique_dataset.csv").values[:,-1]
            for dir in glob.glob(f"{sub_dir}*"):
                self.image_dir_list.append(dir)
                class_name = dir.split("/")[-1]
                self.class_list.append(class_name)
            for sub_image_dir in self.image_dir_list:
                name_list = sorted(os.listdir(sub_image_dir))
                self.image_list += [sub_image_dir + '/' + image_name for image_name in name_list if image_name in unique_images_list]
            print(len(self.image_list))
        else:
            self.label2index_file = image_dir + split + ('/' if len(split) else '') + label2index_file
            self.class_list = ["center"]
            for class_name in self.class_list:
                name_list = sorted(os.listdir(self.image_dir + '/' + class_name))
                self.image_list += [self.image_dir + '/' + class_name + '/' + image_name for image_name in name_list]


        self.label2index_dict = {}
        with open(self.label2index_file, 'r') as f:
            reader = csv.DictReader(f, fieldnames=['frame_id', 'steering_angle'], delimiter=",")
            header = reader.__next__()
            for row in reader:
                self.label2index_dict[row['frame_id']] = row['steering_angle']

        # name_list = sorted(os.listdir(self.image_dir))[:self.args.num_per_class]
        # self.image_list += [self.image_dir + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2]  # label name
        label = self.class_list.index(label)
        label = torch.LongTensor([label]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, label

    def label2index(self, label_name):
        return self.label2index_dict[label_name]




class DRIVENET_Dataset(Dataset):
    def __init__(self,
                 image_dir,
                 split):
        super(DRIVENET_Dataset).__init__()
        self.total_class_num = 1
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((66, 200)),
            transforms.ToTensor(),
        ])

        self.image_list = []

        name_list = sorted(os.listdir(self.image_dir + '/' + split))
        self.image_list += [self.image_dir + '/' + split + '/' + image_name for image_name in name_list]

        # print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = 1
        label = torch.LongTensor([label]).squeeze()

        # image = Image.open(image_path).convert('RGB')
        # image = self.transform(image)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device('cpu')
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[65:-25, :, :]
        image = self.transform(frame)
        # image = frame.unsqueeze(0).to(device)

        return image, label


class DAVE_ORIG2_Dataset(Dataset):
    def __init__(self,
                 image_dir,
                 split):
        super(DAVE_ORIG2_Dataset).__init__()
        self.total_class_num = 1
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((120, 320)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])

        ])

        self.image_list = []

        name_list = sorted(os.listdir(self.image_dir + '/' + split))
        self.image_list += [self.image_dir + '/' + split + '/' + image_name for image_name in name_list]

        # print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = 1
        label = torch.LongTensor([label]).squeeze()

        # image = Image.open(image_path).convert('RGB')
        # image = self.transform(image)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device('cpu')
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[65:-25, :, :]
        image = self.transform(frame)
        # image = frame.unsqueeze(0).to(device)

        return image, label


class CIFAR10Dataset(Dataset):
    def __init__(self,
                 args,
                 image_dir=constants.CIFAR10_JPEG_DIR,
                 split='train'):
        super(CIFAR10Dataset).__init__()
        assert split in ['train', 'test']
        self.total_class_num = 10
        self.args = args
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
            transforms.Resize(self.args.image_size),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

        self.image_list = []
        self.class_list = sorted(os.listdir(self.image_dir))[:self.args.num_class]
        for class_name in self.class_list:
            name_list = sorted(os.listdir(self.image_dir + class_name))[:self.args.num_per_class]
            self.image_list += [self.image_dir + class_name + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2]  # label name
        label = self.class_list.index(label)
        label = torch.LongTensor([label]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, label


class ImageNetDataset(Dataset):
    def __init__(self,
                 args,
                 image_dir=constants.IMAGENET_JPEG_DIR,
                 label2index_file=constants.IMAGENET_LABEL_TO_INDEX,
                 split='train'):
        super(ImageNetDataset).__init__()
        assert split in ['train', 'val']
        self.total_class_num = 1000
        self.args = args
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
            transforms.Resize(self.args.image_size),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.image_list = []

        with open(label2index_file, 'r') as f:
            self.label2index = json.load(f)

        self.class_list = sorted(os.listdir(self.image_dir))[:self.args.num_class]
        for class_name in self.class_list:
            name_list = sorted(os.listdir(self.image_dir + class_name))[:self.args.num_per_class]
            self.image_list += [self.image_dir + class_name + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2]  # label name
        index = self.label2index[label]
        index = torch.LongTensor([index]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index


class IMDBDataset(Dataset):
    def __init__(self,
                 args,
                 split='train'):
        super(IMDBDataset).__init__()
        assert split in ['train', 'test']
        self.total_class_num = 2
        self.args = args

        MAX_VOCAB_SIZE = 25_000
        TEXT = data.Field(tokenize='spacy',
                          tokenizer_language='en_core_web_sm',
                          include_lengths=False,  # set it as true when training
                          fix_length=constants.PAD_LENGTH,
                          batch_first=False)
        LABEL = data.LabelField(dtype=torch.float)
        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
        TEXT.build_vocab(train_data,
                         max_size=MAX_VOCAB_SIZE,
                         vectors='glove.6B.100d',
                         unk_init=torch.Tensor.normal_)
        LABEL.build_vocab(train_data)

        train_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, test_data),
            batch_size=1,
            sort_within_batch=True,
            shuffle=False
        )

        self.text_list = []
        self.label_list = []
        iterator = train_iterator if split == 'train' else test_iterator
        for batch in train_iterator:
            # self.text_list.append(batch.text.squeeze(-1))
            # self.label_list.append(batch.label.squeeze(-1))
            text, text_lengths = batch.text
            self.text_list.append(text)
            self.label_list.append(batch.label)

        print('Total %d Data.' % len(self.text_list))

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, index):
        text = self.text_list[index]
        label = self.label_list[index]
        return text, label


class DataLoader(object):
    def __init__(self, args):
        self.args = args
        self.init_param()

    def init_param(self):
        self.gpus = 1
        # self.gpus = torch.cuda.device_count()
        # TODO: multi GPU

    def get_loader(self, dataset, shuffle=True):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size * self.gpus,
            num_workers=int(self.args.num_workers),
            shuffle=shuffle
        )
        return data_loader


def get_loader(args):
    assert args.dataset in ['CIFAR10', 'ImageNet', 'IMDB', 'Udacity', "dave2", "transfer", "DriverNet"]
    if args.dataset == 'Udacity':
        train_data = UDACITYDataset(args, split='train')
        test_data = UDACITYDataset(args, split='test')
        seed_data = UDACITYDataset(args, image_dir=constants.UDACITY_SEED_DIR,
                                   label2index_file=constants.UDACITY_SEED_LABEL_TO_INDEX, split='seed')
        loader = DataLoader(args)
        train_loader = loader.get_loader(train_data, False)
        test_loader = loader.get_loader(test_data, False)
        seed_loader = loader.get_loader(seed_data, False)
        TOTAL_CLASS_NUM = 1
    elif args.dataset in ['dave2', 'transfer']:
        data = DAVE_ORIG2_Dataset(image_dir=args.image_dir, split=args.split)
        loader = DataLoader(args)
        data_loader = loader.get_loader(data, False)
        return data_loader
    elif args.dataset == 'DriverNet':
        data = DRIVENET_Dataset(image_dir=args.image_dir, split=args.split)
        loader = DataLoader(args)
        data_loader = loader.get_loader(data, False)
        return data_loader
    elif args.dataset == 'CIFAR10':
        train_data = CIFAR10Dataset(args, split='train')
        test_data = CIFAR10Dataset(args, split='test')
        loader = DataLoader(args)
        train_loader = loader.get_loader(train_data, False)
        test_loader = loader.get_loader(test_data, False)
        seed_loader = loader.get_loader(test_data, True)
        TOTAL_CLASS_NUM = 10
    elif args.dataset == 'ImageNet':
        train_data = CIFAR10Dataset(args, split='train')
        test_data = CIFAR10Dataset(args, split='val')
        loader = DataLoader(args)
        train_loader = loader.get_loader(train_data, False)
        test_loader = loader.get_loader(test_data, False)
        seed_loader = loader.get_loader(test_data, True)
        TOTAL_CLASS_NUM = 1000
    elif args.dataset == 'IMDB':
        MAX_VOCAB_SIZE = 25_000
        TEXT = data.Field(tokenize='spacy',
                          tokenizer_language='en_core_web_sm',
                          include_lengths=False,  # set it as true when training
                          fix_length=constants.PAD_LENGTH,
                          batch_first=False)
        LABEL = data.LabelField(dtype=torch.float)
        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
        TEXT.build_vocab(train_data,
                         max_size=MAX_VOCAB_SIZE,
                         vectors='glove.6B.100d',
                         unk_init=torch.Tensor.normal_)
        LABEL.build_vocab(train_data)

        train_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, test_data),
            batch_size=args.batch_size,
            sort_within_batch=True,
            shuffle=False
        )
        _, seed_iterator = data.BucketIterator.splits(
            (train_data, test_data),
            batch_size=args.batch_size,
            sort_within_batch=True,
            shuffle=True
        )
        train_loader = []
        test_loader = []
        seed_loader = []
        for batch in train_iterator:
            train_loader.append((batch.text, batch.label))
        for batch in test_iterator:
            test_loader.append((batch.text, batch.label))
        for batch in seed_iterator:
            seed_loader.append((batch.text, batch.label))
        TOTAL_CLASS_NUM = 2
    return TOTAL_CLASS_NUM, train_loader, test_loader, seed_loader


class FuzzDataset:
    def __init__(self):
        raise NotImplementedError

    def label2index(self):
        raise NotImplementedError

    def get_len(self):
        return len(self.image_list)

    def get_item(self, index):
        image_path = self.image_list[index]
        # label = image_path.split('/')[-2] # label name
        label = image_path.split('/')[-1]  # label name
        # label = self.cat_list.index(label)
        index = float(self.label2index(label))
        # assert int(index) < self.args.num_class
        index = torch.LongTensor([index]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return (image, index)

    def build(self):
        image_list = []
        label_list = []
        # for i in tqdm(range(self.get_len())):
        for i in range(self.get_len()):
            (image, label) = self.get_item(i)
            image_list.append(image)
            label_list.append(label)
        return image_list, label_list

    def to_numpy(self, image_list, is_image=True):
        image_numpy_list = []
        for i in tqdm(range(len(image_list))):
            image = image_list[i]
            if is_image:
                image_numpy = image.transpose(0, 2).numpy()
            else:
                image_numpy = image.numpy()
            image_numpy_list.append(image_numpy)
        print('Numpy: %d' % len(image_numpy_list))
        return image_numpy_list

    def to_batch(self, data_list, is_image=True):
        batch_list = []
        batch = []
        for i, data in enumerate(data_list):
            if i and i % self.args.batch_size == 0:
                batch_list.append(torch.stack(batch, 0))
                batch = []
            batch.append(self.norm(data) if is_image else data)
        if len(batch):
            batch_list.append(torch.stack(batch, 0))
        print('Batch: %d' % len(batch_list))
        return batch_list


class UDACITYFuzzDataset(FuzzDataset):
    def __init__(self,
                 args,
                 image_dir=constants.UDACITY_SEED_DIR,
                 label2index_file=constants.UDACITY_SEED_LABEL_TO_INDEX,
                 split=''):
        self.args = args
        self.label2index_file = image_dir + label2index_file
        self.image_dir = image_dir + split + ('/' if len(split) else '')

        self.transform = transforms.Compose([
            transforms.Resize(self.args.image_size),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
        ])
        self.norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        self.image_list = []
        self.label2index_dict = {}
        with open(self.label2index_file, 'r') as f:
            reader = csv.DictReader(f, fieldnames=['frame_id', 'steering_angle'], delimiter=",")
            header = reader.__next__()
            for row in reader:
                self.label2index_dict[row['frame_id']] = row['steering_angle']

        self.class_list = sorted(os.listdir(self.image_dir))[:self.args.num_class]
        # for class_name in self.class_list:
        #     name_list = sorted(os.listdir(self.image_dir + class_name))[:self.args.num_per_class]
        #     self.image_list += [self.image_dir + class_name + '/' + image_name for image_name in name_list]

        name_list = sorted(os.listdir(self.image_dir))[:self.args.num_per_class]
        self.image_list += [self.image_dir + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def label2index(self, label_name):
        return self.label2index_dict[label_name]


class CIFAR10FuzzDataset(FuzzDataset):
    def __init__(self,
                 args,
                 image_dir=constants.CIFAR10_JPEG_DIR,
                 split='test'):
        self.args = args
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
            transforms.Resize(self.args.image_size),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
        ])
        self.norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        self.image_list = []

        self.class_list = sorted(os.listdir(self.image_dir))[:self.args.num_class]
        for class_name in self.class_list:
            name_list = sorted(os.listdir(self.image_dir + class_name))[:self.args.num_per_class]
            self.image_list += [self.image_dir + class_name + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def label2index(self, label_name):
        return self.class_list.index(label_name)


class ImageNetFuzzDataset(FuzzDataset):
    def __init__(self,
                 args,
                 image_dir=constants.IMAGENET_JPEG_DIR,
                 label2index_file=constants.IMAGENET_LABEL_TO_INDEX,
                 split='val'):
        self.args = args
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
            transforms.Resize(self.args.image_size),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
        ])
        self.norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.image_list = []

        with open(label2index_file, 'r') as f:
            self.label2index_dict = json.load(f)

        self.class_list = sorted(os.listdir(self.image_dir))[:self.args.num_class]
        for class_name in self.class_list:
            name_list = sorted(os.listdir(self.image_dir + class_name))[:self.args.num_per_class]
            self.image_list += [self.image_dir + class_name + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def label2index(self, label_name):
        return self.label2index_dict[label_name]


if __name__ == '__main__':
    pass
import sys
import copy
import time

from models.configs import bcolors
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from models.extract_vgg_layer import FeatureVisualization
import cv2
import math
import random
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import pickle


model_vgg16 = VGG16(weights='imagenet', include_top=False)
def ExtractImageFeature(img_path, aug_features=None, normalization=False):
    conv_3_2 = FeatureVisualization(12)
    executed_image = cv2.imread(img_path)
    features = conv_3_2.save_feature_to_img(executed_image, aug_features)

    return features

def preprocess(path, target_size):
    return preprocess_image(path, target_size)[0]

def data_generator(xs, ys, infos, target_size, batch_size=64):
    gen_state = 0
    while 1:
        if gen_state + batch_size > len(xs):
            paths = xs[gen_state: len(xs)]
            y = ys[gen_state: len(xs)]
            X = [preprocess(x, target_size) for x in paths]
            info = infos[gen_state: len(xs)]
            gen_state = 0
        else:
            paths = xs[gen_state: gen_state + batch_size]
            y = ys[gen_state: gen_state + batch_size]
            X = [preprocess(x, target_size) for x in paths]
            info = infos[gen_state: gen_state + batch_size]
            gen_state += batch_size
        yield np.array(X), np.array(y), np.array(info)


def load_train_data(path='./training/', batch_size=64, shape=(100, 100)):
    xs = []
    ys = []
    start_load_time = time.time()
    with open(path + 'labels.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            xs.append(path + line.split(',')[0])
            ys.append(float(line.split(',')[1]))
    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)

    train_xs = xs
    train_ys = ys

    train_generator = data_generator(train_xs, train_ys,
                                     target_size=(shape[0], shape[1]),
                                     batch_size=batch_size)

    print(bcolors.OKBLUE + 'finished loading data, running time: {} seconds'.format(
        time.time() - start_load_time) + bcolors.ENDC)
    return train_generator, len(train_xs)


def load_test_data(path='./testing/', batch_size=64, shape=(100, 100)):
    xs = []
    ys = []
    infos = []
    start_load_time = time.time()
    with open(path + 'labels.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            xs.append(path + 'center/' + line.split(',')[0] + '.jpg')
            ys.append(float(line.split(',')[1]))
            infos.append(line.split(',')[0])
    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)

    train_xs = xs
    train_ys = ys

    train_generator = data_generator(train_xs, train_ys, infos,
                                     target_size=(shape[0], shape[1]),
                                     batch_size=batch_size)

    print(bcolors.OKBLUE + 'finished loading data, running time: {} seconds'.format(
        time.time() - start_load_time) + bcolors.ENDC)
    return train_generator, len(train_xs)

"""
Provides some useful functions.
"""
def readable_time_str():
    """Get readable time string based on current local time.

    The time string will be formatted as %Y-%m-%d %H:%M:%S.

    Returns
    -------
    str
        Readable time string.

    """
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def user_home_dir():
    """Get path of user home directory.

    Returns
    -------
    str
        Path of user home directory.

    """
    return os.path.expanduser("~")


def to_numpy(data):
    """Convert other data type to numpy. If the data itself
    is numpy type, then a copy will be made and returned.

    Returns
    -------
    numpy.array
        Numpy array of passed data.

    """
    if 'mxnet' in str(type(data)):
        data = data.asnumpy()
    elif 'torch' in str(type(data)):
        data = data.cpu().numpy()
    elif 'numpy' in str(type(data)):
        data = np.copy(data)
    return data


def save_pickle(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data

def save_npy(path, data):
    with open(path, 'wb') as f:
        np.save(f, np.array(data))


def load_npy(path):
    with open(path, 'rb') as f:
        data = np.load(f)
    return data

def draw_arrow(img, angle1, angle2, angle3):
    pt1 = (img.shape[1] / 2, img.shape[0])
    pt2_angle1 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle1)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle1)))
    pt2_angle2 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle2)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle2)))
    pt2_angle3 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle3)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle3)))
    img = cv2.arrowedLine(img, pt1, pt2_angle1, (0, 0, 255), 1)
    img = cv2.arrowedLine(img, pt1, pt2_angle2, (0, 255, 0), 1)
    img = cv2.arrowedLine(img, pt1, pt2_angle3, (255, 0, 0), 1)
    return img


def angle_diverged(angle1, angle2, angle3):
    if (abs(angle1 - angle2) > 0.2 or abs(angle1 - angle3) > 0.2 or abs(angle2 - angle3) > 0.2) and not (
                (angle1 > 0 and angle2 > 0 and angle3 > 0) or (
                                angle1 < 0 and angle2 < 0 and angle3 < 0)):
        return True
    return False


def preprocess_image(img_path, target_size=(100, 100)):
    img = image.load_img(img_path, target_size=target_size)
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    return input_img_data


def deprocess_image(x):
    x = x.reshape((100, 100, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def atan_layer(x):
    return tf.multiply(tf.atan(x), 2)


def atan_layer_shape(input_shape):
    return input_shape


def normal_init(shape):
    return K.truncated_normal(shape, stddev=0.1)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = 500 * np.mean(gradients)
    return grad_mean * new_grads


def constraint_black(gradients, rect_shape=(10, 10)):
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads


def init_coverage_tables(model1, model2, model3):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    init_dict(model3, model_layer_dict3)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3


def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index


def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def update_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in range(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True


def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True


def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    if not predictions1 == predictions2 == predictions3:
        return True
    return False

class Logger(object):
    def __init__(self, *files):
        self.files = files

    def open(self):
        """ Redirect stdout """
        if not hasattr(sys, '_stdout'):
            # Only do this once just in case stdout was already initialized
            # @note Will fail if stdout for some reason changes
            sys._stdout = sys.stdout
        sys.stdout = self
        return self

    def close(self):
        """ Restore """
        stdout = sys._stdout
        for f in self.files:
            if f != stdout:
                f.close()
        sys.stdout = stdout

    def write(self, obj):
        for f in self.files:
            f.write(obj)


def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s
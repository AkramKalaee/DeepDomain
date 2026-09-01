from random import randint as rd
import numpy as np
import torch
import math
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pathlib
import torch.nn as nn
# from torch import nn.torch.flatten()
import torch.nn.functional as F
# import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Convolution2D, Dense, Flatten, Lambda, MaxPooling2D, Dropout, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from constants import SRC_DIR
from models.utils.common import load_test_data, ExtractImageFeature, save_npy, load_npy
from sklearn.decomposition import PCA as RandomizedPCA
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# ###################
# # config
# ###################
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ###################
# # some utils
# ###################
def atan_layer(x):
    return tf.multiply(tf.atan(x), 2)

def atan_layer_shape(input_shape):
    return input_shape


def normal_init(shape):
    # return K.truncated_normal(shape, stddev=0.1)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return K.variable(initial)


# ###################
# # dave_orig
# ###################
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

class dave_orig(nn.Module):
    def __init__(self):
        super(dave_orig, self).__init__()
        conv1 = nn.Conv2d(3, 24, 5, stride=2)
        nn.init.zeros_(conv1.weight)
        nn.init.zeros_(conv1.bias)
        conv2 = nn.Conv2d(24, 36, 5, stride=2)
        nn.init.zeros_(conv2.weight)
        nn.init.zeros_(conv2.bias)
        conv3 =  nn.Conv2d(36, 48, 5, stride=2)
        nn.init.zeros_(conv2.weight)
        nn.init.zeros_(conv2.bias)
        conv4 = nn.Conv2d(48, 64, 3)
        nn.init.zeros_(conv2.weight)
        nn.init.zeros_(conv2.bias)
        conv5 = nn.Conv2d(64, 64, 3)
        nn.init.zeros_(conv2.weight)
        nn.init.zeros_(conv2.bias)

        self.conv_layers = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            conv3,
            nn.ReLU(),
            conv4,
            nn.ReLU(),
            conv5,
            nn.ReLU(),
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=1600, out_features=1164),
            nn.ReLU(),
            nn.Linear(in_features=1164, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, data):
        # data = data.reshape(data.size(0), 1, 60, 120)
        # print(data.shape)
        output = self.conv_layers(data)
        # print(output.shape)
        output = output.contiguous().view(output.size(0), -1)
        # print(output.shape)
        output = self.dense_layers(output)
        return output

def dave_orig_keras(input_tensor=None, load_weights=False, weights_path=None):  # original dave
    if input_tensor is None:
        input_tensor = Input(shape=(100, 100, 3))
    x = Convolution2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2), name='block1_conv1',
                      data_format='channels_last',
                      kernel_initializer='zeros',
                      bias_initializer='zeros'
                      )(input_tensor)
    x = Convolution2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2), name='block1_conv2',
                      data_format='channels_last',
                      kernel_initializer='zeros',
                      bias_initializer='zeros'
                      )(x)
    x = Convolution2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2), name='block1_conv3',
                      data_format='channels_last',
                      kernel_initializer='zeros',
                      bias_initializer='zeros'
                      )(x)
    x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1), name='block1_conv4',
                      data_format='channels_last',
                      kernel_initializer='zeros',
                      bias_initializer='zeros'
                      )(x)
    x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1), name='block1_conv5',
                      data_format='channels_last',
                      kernel_initializer='zeros',
                      bias_initializer='zeros'
                      )(x)
    x = Flatten(name='flatten')(x)

    x = Dense(1164, activation='relu', name='fc1')(x)
    x = Dense(100, activation='relu', name='fc2')(x)
    x = Dense(50, activation='relu', name='fc3')(x)
    x = Dense(10, activation='relu', name='fc4')(x)
    x = Dense(1, name='before_prediction')(x)
    x = Lambda(atan_layer, output_shape=atan_layer_shape, name='prediction')(x)

    m = Model(input_tensor, x)
    if load_weights:
        m.load_weights(weights_path)

    # compiling
    m.compile(loss='mse', optimizer='adadelta')
    print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
    return m


def dave_orig_v2_keras(input_tensor=None, y=None, load_weights=False, weights_path=None):  # original dave
    if input_tensor is None:
        input_tensor = Input(shape=(100, 100, 3))
        y = Input((1,), name='y', dtype=tf.float32)

    x = Convolution2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2), name='block1_conv1')(input_tensor)
    x = Convolution2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2), name='block1_conv2')(x)
    x = Convolution2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2), name='block1_conv3')(x)
    x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1), name='block1_conv4')(x)
    x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1), name='block1_conv5')(x)
    x = Flatten(name='flatten')(x)

    x = Dense(1164, activation='relu', name='fc1')(x)
    x = Dense(100, activation='relu', name='fc2')(x)
    x = Dense(50, activation='relu', name='fc3')(x)
    x = Dense(10, activation='relu', name='fc4')(x)
    # x = Dense(2, name='before_prediction')(x)
    # x = Lambda(atan_layer, output_shape=atan_layer_shape, name='prediction')(x)
    #version 1
    mu = Dense(1, name='before_prediction')(x)
    mu = Lambda(atan_layer, output_shape=atan_layer_shape, name='prediction')(mu)

    sigma = Dense(1, activation=lambda i_x: tf.nn.elu(i_x) + 1)(x)
    #version 2
    # x = Dense(2)(x) # Output = (μ, ln(σ))

    # compiling
    # def loss(y_true, y_pred):
    #     mu = y_pred[:, :1]  # first output neuron
    #     log_sig = y_pred[:, 1:]  # second output neuron
    #     sig = tf.exp(log_sig)  # undo the log
    #
    #     return tf.reduce_mean(2 * log_sig + ((y_true - mu) / sig) ** 2)

    def mdn_cost(mu, sigma, y):
        dist = tfd.Normal(loc=mu, scale=sigma)
        return tf.reduce_mean(-dist._log_prob(y))



    m = Model(inputs=[input_tensor, y], outputs=[mu, sigma])
    if load_weights:
        m.load_weights(weights_path)

    m.compile(loss= mdn_cost(mu, sigma, y), optimizer='adadelta')
    print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
    return m

def train_dave(model_name):
    # train the model
    batch_size = 256
    nb_epoch = 10
    # model_name = sys.argv[1]

    if model_name == '1':
        model = dave_orig_v2_keras()
        save_model_name = f'{SRC_DIR}/models/dave_orig/dave_orig_v2.h5'
    elif model_name == '2':
        # K.set_learning_phase(1)
        model = Dave_norminit()
        save_model_name = './Model2.h5'
    elif model_name == '3':
        # K.set_learning_phase(1)
        model = Dave_dropout()
        save_model_name = './Model3.h5'
    else:
        print(bcolors.FAIL + 'invalid model name, must one of 1, 2 or 3' + bcolors.ENDC)

    # the data, shuffled and split between train and test sets
    training_path = f"{SRC_DIR}/dataset/train/"
    train_generator, samples_per_epoch = load_train_data(path=training_path, batch_size=batch_size, shape=(100, 100))

    # trainig
    model.fit_generator(train_generator,
                        steps_per_epoch=math.ceil(samples_per_epoch * 1. / batch_size),
                        epochs=nb_epoch,
                        workers=0,
                        use_multiprocessing=True)
    print(bcolors.OKGREEN + 'Model trained' + bcolors.ENDC)

    # evaluation
    K.set_learning_phase(0)
    testing_path = f"{SRC_DIR}/dataset/test/"
    test_generator, samples_per_epoch = load_test_data(path=testing_path, batch_size=batch_size, shape=(100, 100))
    model.evaluate_generator(test_generator,
                             steps=math.ceil(samples_per_epoch * 1. / batch_size))
    # save model
    model.save_weights(save_model_name)

def create_dave_orig_torch_weights(model_keras, model_pyt, weights_path):
    model_pyt.conv_layers[0].weight.data = torch.tensor(model_keras.layers[1].get_weights()[0].T)
    model_pyt.conv_layers[0].bias.data = torch.tensor(model_keras.layers[1].get_weights()[1])

    model_pyt.conv_layers[2].weight.data = torch.tensor(model_keras.layers[2].get_weights()[0].T)
    model_pyt.conv_layers[2].bias.data = torch.tensor(model_keras.layers[2].get_weights()[1])

    model_pyt.conv_layers[4].weight.data = torch.tensor(model_keras.layers[3].get_weights()[0].T)
    model_pyt.conv_layers[4].bias.data = torch.tensor(model_keras.layers[3].get_weights()[1])

    model_pyt.conv_layers[6].weight.data = torch.tensor(model_keras.layers[4].get_weights()[0].T)
    model_pyt.conv_layers[6].bias.data = torch.tensor(model_keras.layers[4].get_weights()[1])

    model_pyt.conv_layers[8].weight.data = torch.tensor(model_keras.layers[5].get_weights()[0].T)
    model_pyt.conv_layers[8].bias.data = torch.tensor(model_keras.layers[5].get_weights()[1])


    model_pyt.dense_layers[0].weight.data = torch.tensor(model_keras.layers[7].get_weights()[0].T)
    model_pyt.dense_layers[0].bias.data = torch.tensor(model_keras.layers[7].get_weights()[1])

    model_pyt.dense_layers[2].weight.data = torch.tensor(model_keras.layers[8].get_weights()[0].T)
    model_pyt.dense_layers[2].bias.data = torch.tensor(model_keras.layers[8].get_weights()[1])

    model_pyt.dense_layers[4].weight.data = torch.tensor(model_keras.layers[9].get_weights()[0].T)
    model_pyt.dense_layers[4].bias.data = torch.tensor(model_keras.layers[9].get_weights()[1])

    model_pyt.dense_layers[6].weight.data = torch.tensor(model_keras.layers[10].get_weights()[0].T)
    model_pyt.dense_layers[6].bias.data = torch.tensor(model_keras.layers[10].get_weights()[1])

    torch.save(model_pyt.state_dict(), weights_path)

    # print(model_keras.layers[10].weights)
    # print(np.transpose(model_pyt.dense_layers[9].weight.data))

# ###################
# # dave_norminit
# ###################
class dave_norminit(nn.Module):
    def __init__(self):
        super(dave_norminit, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
        )
        self.dense_layers = nn.Sequential(
            # nn.BatchNorm1d(1600),
            nn.Linear(in_features=1152, out_features=1164),
            nn.ReLU(),
            # nn.BatchNorm1d(1164),
            nn.Linear(in_features=1164, out_features=100),
            nn.ReLU(),
            # nn.BatchNorm1d(100),
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(),
            # nn.BatchNorm1d(50),
            nn.Linear(in_features=50, out_features=10),
            nn.ReLU(),
            # nn.BatchNorm1d(10),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, data):
        data = data.reshape(data.size(0), 3, 66, 200)
        # print(data.shape)
        output = self.conv_layers(data)
        # print(output.shape)
        output = output.view(output.size(0), -1)
        # print(output.shape)
        output = self.dense_layers(output)
        return output
def dave_norminit_keras(input_tensor=None, load_weights=False, weights_path=None):  # original dave with normal initialization
    if input_tensor is None:
        input_tensor = Input(shape=(66, 200, 3))
    x = Convolution2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2),
                      name='block1_conv1')(input_tensor)
    x = Convolution2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2),
                      name='block1_conv2')(x)
    x = Convolution2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2),
                      name='block1_conv3')(x)
    x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1),
                      name='block1_conv4')(x)
    x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1),
                      name='block1_conv5')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(1164, activation='relu', name='fc1')(x) #kernel_initializer=normal_init,
    x = Dense(100, activation='relu', name='fc2')(x)
    x = Dense(50, activation='relu', name='fc3')(x)
    x = Dense(10, activation='relu', name='fc4')(x)
    x = Dense(1, name='before_prediction')(x)
    x = Lambda(atan_layer, output_shape=atan_layer_shape, name='prediction')(x)

    m = Model(input_tensor, x)
    if load_weights:
        m.load_weights(weights_path)

    # compiling
    m.compile(loss='mse', optimizer='Adam')
    print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
    return m
def create_dave_norminit_torch_weights(model_keras, model_pyt, weights_path):
    model_pyt.conv_layers[0].weight.data = torch.tensor(model_keras.layers[1].get_weights()[0].T)
    model_pyt.conv_layers[0].bias.data = torch.tensor(model_keras.layers[1].get_weights()[1])

    model_pyt.conv_layers[2].weight.data = torch.tensor(model_keras.layers[2].get_weights()[0].T)
    model_pyt.conv_layers[2].bias.data = torch.tensor(model_keras.layers[2].get_weights()[1])

    model_pyt.conv_layers[4].weight.data = torch.tensor(model_keras.layers[3].get_weights()[0].T)
    model_pyt.conv_layers[4].bias.data = torch.tensor(model_keras.layers[3].get_weights()[1])

    model_pyt.conv_layers[6].weight.data = torch.tensor(model_keras.layers[4].get_weights()[0].T)
    model_pyt.conv_layers[6].bias.data = torch.tensor(model_keras.layers[4].get_weights()[1])

    model_pyt.conv_layers[8].weight.data = torch.tensor(model_keras.layers[5].get_weights()[0].T)
    model_pyt.conv_layers[8].bias.data = torch.tensor(model_keras.layers[5].get_weights()[1])


    model_pyt.dense_layers[0].weight.data = torch.tensor(model_keras.layers[7].get_weights()[0].T)
    model_pyt.dense_layers[0].bias.data = torch.tensor(model_keras.layers[7].get_weights()[1])

    model_pyt.dense_layers[2].weight.data = torch.tensor(model_keras.layers[8].get_weights()[0].T)
    model_pyt.dense_layers[2].bias.data = torch.tensor(model_keras.layers[8].get_weights()[1])

    model_pyt.dense_layers[4].weight.data = torch.tensor(model_keras.layers[9].get_weights()[0].T)
    model_pyt.dense_layers[4].bias.data = torch.tensor(model_keras.layers[9].get_weights()[1])

    model_pyt.dense_layers[6].weight.data = torch.tensor(model_keras.layers[10].get_weights()[0].T)
    model_pyt.dense_layers[6].bias.data = torch.tensor(model_keras.layers[10].get_weights()[1])

    torch.save(model_pyt.state_dict(), weights_path)

    # print(model_keras.layers[10].weights)
    # print(np.transpose(model_pyt.dense_layers[9].weight.data))
# ###################
# # dave_dropout
# ###################
class dave_dropout(nn.Module):
    def __init__(self):
        super(dave_dropout, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=1792, out_features=500), #4096
            nn.Dropout(0.5),
            nn.Linear(in_features=500, out_features=100),
            nn.Dropout(0.25),
            nn.Linear(in_features=100, out_features=20),
            nn.Linear(in_features=20, out_features=1)
        )

    def forward(self, data):
        data = data.reshape(data.size(0), 3, 32, 128)
        # print(data.shape)
        output = self.conv_layers(data)
        # print(output.shape)
        output = output.view(output.size(0), -1)
        # print(output.shape)
        output = self.dense_layers(output)
        return output


def dave_dropout_keras(input_tensor=None, load_weights=False, weights_path=None):  # original dave
    if input_tensor is None:
        input_tensor = Input(shape=(32, 128, 3))

    x = Convolution2D(16, (3, 3), activation='relu', padding='valid', strides=(1, 1),
                      name='convolution2d_1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), padding='valid', strides=(2, 2), data_format='channels_last',
                     name="maxpooling2d_1")(x)
    x = Convolution2D(32, (3, 3), activation='relu', padding='valid', strides=(1, 1),
                      name='convolution2d_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='valid', strides=(2, 2), data_format='channels_last',
                     name="maxpooling2d_2")(x)
    x = Convolution2D(64, (3, 3), activation='relu', padding='valid', strides=(1, 1),
                      name='convolution2d_3')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='valid', strides=(2, 2), data_format='channels_last',
                     name="maxpooling2d_3")(x)
    x = Flatten(name='flatten_1')(x)

    x = Dense(500, activation='relu', name='dense_1')(x)
    x = Dropout(rate=.5, name='dropout_1')(x, training=True)
    x = Dense(100, activation='relu', name='dense_2')(x)
    x = Dropout(rate=.25, name='dropout_2')(x, training=True)
    x = Dense(20, activation='relu', name='dense_3')(x)
    x = Dense(1, activation='linear', name='dense_4')(x)

    m = Model(input_tensor, x)

    if load_weights:
        m.load_weights(weights_path)
    opt = Adam(learning_rate=1e-04)
    m.compile(loss='mean_squared_error', optimizer=opt)

    return m
def create_dave_dropout_torch_weights(model_keras, model_pyt, weights_path):
    model_pyt.conv_layers[0].weight.data = torch.tensor(model_keras.layers[1].get_weights()[0].T)
    model_pyt.conv_layers[0].bias.data = torch.tensor(model_keras.layers[1].get_weights()[1])

    model_pyt.conv_layers[3].weight.data = torch.tensor(model_keras.layers[3].get_weights()[0].T)
    model_pyt.conv_layers[3].bias.data = torch.tensor(model_keras.layers[3].get_weights()[1])

    model_pyt.conv_layers[6].weight.data = torch.tensor(model_keras.layers[5].get_weights()[0].T)
    model_pyt.conv_layers[6].bias.data = torch.tensor(model_keras.layers[5].get_weights()[1])

    model_pyt.dense_layers[0].weight.data = torch.tensor(model_keras.layers[8].get_weights()[0].T)
    model_pyt.dense_layers[0].bias.data = torch.tensor(model_keras.layers[8].get_weights()[1])

    model_pyt.dense_layers[2].weight.data = torch.tensor(model_keras.layers[10].get_weights()[0].T)
    model_pyt.dense_layers[2].bias.data = torch.tensor(model_keras.layers[10].get_weights()[1])

    model_pyt.dense_layers[4].weight.data = torch.tensor(model_keras.layers[12].get_weights()[0].T)
    model_pyt.dense_layers[4].bias.data = torch.tensor(model_keras.layers[12].get_weights()[1])

    model_pyt.dense_layers[5].weight.data = torch.tensor(model_keras.layers[13].get_weights()[0].T)
    model_pyt.dense_layers[5].bias.data = torch.tensor(model_keras.layers[13].get_weights()[1])

    torch.save(model_pyt.state_dict(), weights_path)

    # print(model_keras.layers[10].weights)
    # print(np.transpose(model_pyt.dense_layers[9].weight.data))

def test_create_dave_pytoch():
    models = ["dave_orig"]  # ,"dave_norminit","dave_dropout"
    for model in models:
        path = f"{SRC_DIR}/models/{model}/{model}"
        if model == "dave_orig":
            model_pyt = dave_orig()
            m_keras = dave_orig_keras(load_weights=True, weights_path=f"{path}.h5")
            create_dave_orig_torch_weights(m_keras, model_pyt, weights_path=f"{path}.pt")
        elif model == "dave_norminit":
            model_pyt = dave_norminit()
            m_keras = dave_norminit_keras(load_weights=True, weights_path=f"{path}.hdf5")
            create_dave_norminit_torch_weights(m_keras, model_pyt, weights_path=f"{path}.pt")
        else:
            model_pyt = dave_dropout()
            m_keras = dave_dropout_keras(load_weights=True, weights_path=f"{path}.h5")
            create_dave_dropout_torch_weights(m_keras, model_pyt, weights_path=f"{path}.pt")

def test_manifold():
    extract_features = False
    do_pca = True

    batch_size = 256
    model_name = "dave_orig"
    src_path = r"E:\members\kalaee\working\deep-domain-server-evaluation"
    output_path = f"{src_path}\outputs"
    manifolds_path = f'{output_path}/manifolds/{model_name}_manifolds'
    image_embeddings_path = f"{manifolds_path}/image_embeddings_vgg.npy"
    extracted_feature_path = f"{manifolds_path}/extracted_image_features"
    testing_path = f"{SRC_DIR}/dataset/test/"

    test_generator, samples_per_epoch = load_test_data(path=testing_path, batch_size=batch_size, shape=(100, 100))

    #step0: reading images
    print("reading images...")
    image_data = []
    feature_lists = []
    epoch = 1
    flag = False
    for batch_data in test_generator:
        print(f"epoch {epoch}: {epoch * batch_size}/{samples_per_epoch}")
        for i in range(batch_size):
            if epoch * batch_size + i > samples_per_epoch:
                flag = True
                break
            image = batch_data[0][i]
            image_data.append(image)
            # feature_lists.append(np.reshape(image, (1,-1)))

        epoch += 1
        if flag:
            print(f"epoch {epoch}: {samples_per_epoch}/{samples_per_epoch}")
            break

    # image_embeddings = np.vstack(feature_lists)
    # save_npy(image_embeddings_path, image_embeddings)

    #step1: extract features

    if extract_features:
        print("extracting features...")
        pathlib.Path(extracted_feature_path).mkdir(parents=True, exist_ok=True)
        feature_lists = []
        epoch = 1
        flag = False
        for batch_data in test_generator:
            print(f"epoch {epoch}: {epoch*batch_size}/{samples_per_epoch}")
            for i in range(batch_size):
                if epoch * batch_size + i > samples_per_epoch:
                    flag = True
                    break
                image_data = batch_data[0][i]
                image_name = batch_data[2][i]
                extracted_feature = ExtractImageFeature(image_data)
                save_npy(f"{extracted_feature_path}/{image_name}.npy", extracted_feature)

                features_norm = normalize(extracted_feature[:, np.newaxis], axis=0).ravel()
                feature_lists.append(features_norm)
            epoch += 1
            if flag:
                print(f"epoch {epoch}: {samples_per_epoch}/{samples_per_epoch}")
                break


        image_embeddings = np.vstack(feature_lists)
        save_npy(image_embeddings_path, image_embeddings)
    else:
        print("loading features...")
        image_embeddings = load_npy(image_embeddings_path)

    #step2: estimating number of components
    if do_pca:
        print("estimating number of components...")
        model = RandomizedPCA(100).fit(image_embeddings)
        variance = np.cumsum(model.explained_variance_ratio_)
        plt.plot(variance)
        plt.xlabel('n components')
        plt.ylabel('cumulative variance')
        plt.show()

        print(variance[-1])

    #step3: visualization of manifolds
    print("visualization of manifolds...")
    from sklearn.manifold import Isomap
    model = Isomap(n_components=2)
    proj = model.fit_transform(image_embeddings)
    print(proj.shape)
    plt.scatter(proj[:, 0], proj[:, 1], c=target, cmap=plt.cm.get_cmap('jet', 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5);

    from matplotlib import offsetbox

    def plot_components(data, model, images=None, ax=None,
                        thumb_frac=0.05, cmap='gray'):
        ax = ax or plt.gca()

        proj = model.fit_transform(data)
        ax.plot(proj[:, 0], proj[:, 1], '.k')

        if images is not None:
            min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
            shown_images = np.array([2 * proj.max(0)])
            for i in range(data.shape[0]):
                dist = np.sum((proj[i] - shown_images) ** 2, 1)
                if np.min(dist) < min_dist_2:
                    # don't show points that are too close
                    continue
                shown_images = np.vstack([shown_images, proj[i]])
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(images[i], cmap=cmap),
                    proj[i])
                ax.add_artist(imagebox)

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_components(image_embeddings,
                    model=Isomap(n_neighbors=5, n_components=2, eigen_solver='dense'),
                    images=np.array(image_data)[:, ::100, ::100].reshape((-1, 28, 28)), ax=ax)
    plt.show()

if __name__ == "__main__":
     # train_dave(model_name='1') #dave_orig_v2
     # test_manifold()
     test_create_dave_pytoch()

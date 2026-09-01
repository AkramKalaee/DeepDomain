import pickle
import argparse
from tools.DEEPDOMAIN.data_loader import *
import shutil
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def copy_file(source_file_path, destination_dir):
    try:
        shutil.copy(source_file_path, destination_dir)
    except PermissionError:
        print("PermissionError")
    except OSError:
        print("OSError")

def move_file(source_file_path, destination_dir):
    try:
        shutil.move(source_file_path, destination_dir)
    except PermissionError:
        print("PermissionError")
    except OSError:
        print("OSError")

def get_data_loader(image_dir, split, image_size, dataset):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=image_size)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_class', type=float, default=1)
    parser.add_argument('--image_dir', type=str, default=image_dir)
    parser.add_argument('--split', type=str, default=split)
    parser.add_argument('--dataset', type=str, default=dataset)
    args = parser.parse_args()
    data_loader = get_loader(args)
    return data_loader

def get_branch_distance(lhs, rhs, op="lte", domain=(0, 1), k=1):
    def norm(value, domain):
        if domain is not None:
            min_val = domain[0]
            max_val = domain[1]
            return (value - min_val) / (max_val - min_val)
        else:
            return value / (value + 1)

    if op == "lte":
        if lhs <= rhs:
            return 0
    elif op == "lt":
        if lhs < rhs:
            return 0
    elif op == "gte":
        if lhs >= rhs:
            return 0
    elif op == "gt":
        if lhs > rhs:
            return 0
    distance = k * norm(abs(lhs - rhs), domain)
    return distance

def map_float_to_class(yaht, classes_info):
    for index, info in enumerate(classes_info):
        if yaht >= info[0] and yaht < info[1]:
            return index + 2
    return -1

def get_steering_classes(steering_min=-1, steering_max=1, step=0.1):
    classes_info = []
    n = int((steering_max - steering_min) / step)
    s = steering_min
    for i in range(n):
        e = float("{:.2f}".format(s + step))
        classes_info.append((s, e))
        s = e

    return classes_info[1:]

def init_log(path, headres):
    if os.path.exists(path):
        return
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headres
            )

def log_step(row, path):
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def save_image(img, path):
    cv2.imwrite(path, img)


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

def calculate_euclidean_distance(A, B):
    BT = B.transpose()
    vecProd = np.dot(A, BT)
    SqA = A ** 2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED

def plot_clusters(path, data, labels):
    plot_kwds = {'alpha': 0.15, 's': 80, 'linewidths': 0}
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.savefig(path)
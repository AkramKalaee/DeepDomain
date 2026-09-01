import sys

from constants import SRC_DIR

sys.path.insert(0, "/\\")
from datetime import datetime
from ast import literal_eval
from tools.DEEPDOMAIN.utils import *
from models.utils.common import *
from tools.DEEPDOMAIN.pymoo.optimize import minimize
from tools.DEEPDOMAIN.pymoo.indicators.hv import Hypervolume
from tools.DEEPDOMAIN.pymoo.operators.repair.rounding import RoundingRepair
from tools.DEEPDOMAIN.pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from tools.DEEPDOMAIN.pymoo.core.variable import Real, Integer
from tools.DEEPDOMAIN.pymoo.termination.max_eval import MaximumFunctionCallTermination
from tools.DEEPDOMAIN.pymoo.termination.max_time import TimeBasedTermination
from tools.DEEPDOMAIN.pymoo.algorithms.soo.nonconvex.random_search import RandomSearch
from tools.DEEPDOMAIN.pymoo.algorithms.moo.random_search import RandomSearch as multi_RandomSearch
from tools.DEEPDOMAIN.pymoo.visualization.scatter import Scatter
from tools.DEEPDOMAIN.pymoo.core.problem import Problem
from tools.DEEPDOMAIN.pymoo.algorithms.moo.nsga2 import NSGA2
from tools.DEEPDOMAIN.pymoo.operators.crossover.sbx import SBX
from tools.DEEPDOMAIN.pymoo.operators.mutation.pm import PM
from tools.DEEPDOMAIN.pymoo.termination.default import DefaultMultiObjectiveTermination
from tools.DEEPDOMAIN.pymoo.util.running_metric import RunningMetricAnimation
from tools.DEEPDOMAIN.pymoo.core.callback import Callback
from tools.DEEPDOMAIN.pymoo.algorithms.moo.dnsga2 import DNSGA2

from tools.DEEPDOMAIN.pathway_grad.src.Pruner import Pruner

import tools.DEEPDOMAIN.automold.Automold as am
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import torch
import cv2
import csv
import copy
import os
from models.utils.common import save_npy

torch.cuda.empty_cache()

class Objectives:
    def __init__(self, inconsistent_pool_path, save_path, max_n_clauses=1, inconsistents_NBC=None, gama=0.2):
        self.lambda_ = 5
        self.epsilon_ = 0.03
        self.K = 1
        self.delta = 0.5
        self.gama = gama  # [0.2, 0.3, 0.4]
        self.inconsistent_pool_path = inconsistent_pool_path
        self.max_n_clauses = max_n_clauses
        self.save_path = save_path
        self.inconsistents_NBC = inconsistents_NBC

    def f1(self, source_test_case, follow_up_test_case, config):
        follow_up_test_case.extract_pathway(config)
        """
        Structural metric
        """

        # branch distance
        def calculate_branch_distance(threshold, layer_path1, layer_path2, k):
            distance = 0
            n_layers = len(layer_path1)

            for i in range(n_layers):
                # n_neurons = len(layer_path1[i])
                n_neurons = np.where(layer_path1[i] != 0)[0]
                for j in n_neurons:

                    # if layer_path1[i][j] == 0:  # true branch: scores <= threshold
                    #     # op = "lte"
                    #     continue
                    # else:  # true branch: score > threshold
                    #     op = "gt"

                    score = layer_path2[i][j]
                    if score == -1:
                        distance += 1
                        continue

                    lhs = score
                    rhs = threshold
                    dist = get_branch_distance(lhs, rhs, op="gt", domain=None, k=k)
                    distance += dist

            return distance

        # dissimilarity
        def calculate_dissimilarity(layer_path1, layer_path2):
            sum_distance = 0
            n_layers = len(layer_path1)
            n_parameters = 0
            for i in range(n_layers):
                distance = np.abs(np.subtract(layer_path1[i], layer_path2[i]))
                sum_distance += np.sum(distance)
                n_parameters += len(distance)

            return sum_distance, n_parameters

        # version 1: branch distance
        # distances = 0
        # pool_size = 0
        # class_no = follow_up_test_case.pred_class_no
        # if class_no in self.inconsistents_indexer.keys():
        #     indexer = self.inconsistents_indexer[class_no]
        #     index = indexer[0]
        #     critical_layer_paths_list = indexer[1]
        #     remove_threshold_list = indexer[2]
        #
        #     x_test = follow_up_test_case.critical_path
        #     idx = index.search(x_test[np.newaxis], 1)[1].flatten()[0]
        #     remove_threshold = remove_threshold_list[idx]
        #     critical_layer_paths = critical_layer_paths_list[idx]
        #     distance = calculate_branch_distance(remove_threshold,
        #                                   critical_layer_paths,
        #                                   follow_up_test_case.layer_integrad_scores, self.K)
        #
        #     distances += distance
        #     pool_size += 1
        #
        # if pool_size == 0:
        #     fitness1 = 1
        # else:
        #     fitness1 = distances / (pool_size * self.max_n_clauses)

        # version 2: dissimilarity
        # distance, total_parameters = calculate_dissimilarity(source_test_case.critical_layer_paths,
        #                                                 follow_up_test_case.critical_layer_paths)
        #
        # fitness2 = distance / total_parameters
        #
        # fitness = fitness1 + fitness2
        # print(f"--f1 : {fitness}")

        # calculate NBC coverage

        fitness1 = 0
        class_no = follow_up_test_case.pred_class_no
        if class_no in self.inconsistents_NBC.keys():
            nbc = self.inconsistents_NBC[class_no]
            image = follow_up_test_case.image_data
            cache_dir = f"{SRC_DIR}/cache/seeds"
            directory = f"{cache_dir}/{class_no}"
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"{directory}/1.jpg", image)

            test_loader = get_data_loader(cache_dir, str(class_no), follow_up_test_case.image_data.shape, config.model_name)
            criterion1 = copy.deepcopy(nbc)
            criterion1.critical_layer_paths = follow_up_test_case.critical_layer_paths
            criterion1.assess(test_loader)
            fitness1 = criterion1.current
            del criterion1


        # version 1
        distance = calculate_branch_distance(source_test_case.remove_threshold,
                                          source_test_case.critical_layer_paths,
                                          follow_up_test_case.layer_integrad_scores, self.K)

        follow_up_test_case.critical_path = None
        follow_up_test_case.critical_layer_paths = None
        follow_up_test_case.original_layer_paths = None
        follow_up_test_case.layer_integrad_scores = None
        follow_up_test_case.remove_threshold = None

        fitness2 = distance / self.max_n_clauses

        # version 2
        # distance, total_parameters = calculate_dissimilarity(source_test_case.critical_layer_paths,
        #                                                      follow_up_test_case.critical_layer_paths)
        # fitness2 = distance / total_parameters
        # version 3
        # source_point = np.array([follow_up_test_case.critical_path], dtype=np.float64)
        # input_space = np.array([source_test_case.critical_path], dtype=np.float64)
        # distance = calculate_euclidean_distance(source_point, input_space)
        # distance = np.array(distance, dtype=np.float64)
        # distance = np.mean(distance, axis=1, keepdims=True)[0][0]
        #
        # fitness2 = distance / np.sqrt(source_point.shape[1])

        # version 4
        # distance = np.linalg.norm(source_test_case.critical_path - follow_up_test_case.critical_path)
        # fitness2 = distance / np.sqrt(source_test_case.critical_path.shape[0])
        #
        # fitness = fitness1 / (1 - fitness2)

        fitness = fitness1 + fitness2

        return fitness


    def f2(self, source_test_case, follow_up_test_case, config):
        """
        Metamorphic rule
        """
        # ref.: deep_test
        # original_label = source_test_case.label
        original_pred = source_test_case.pred

        consistent = True
        lhs = abs(original_pred - follow_up_test_case.pred)
        rhs = self.gama
        distance = get_branch_distance(lhs, rhs, op="gt", domain=(0, 2 - self.gama), k=self.K)
        if distance == 0:
            consistent = False

        fitness = 1 - distance

        follow_up_test_case.consistent = consistent
        # print(f"--f2 : {fitness}    consistent: {consistent}")

        if not consistent:
            class_no = follow_up_test_case.pred_class_no
            base_name = source_test_case.name
            suffix = datetime.utcnow().strftime("%y%m%d_%H%M%S%f")
            file_name = "_".join([base_name, suffix])
            save_path = f"{self.save_path}/{class_no}"
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            directory = f"{save_path}/{file_name}.jpg"
            copy_file(config.prediction_cache_path+"1.jpg", directory)

        return fitness


class Test_Suite:
    def __init__(self, source_test_case, cache):
        self.source = source_test_case
        self.follow_ups = []
        self.cache = cache
        self.size = None
        self.directory = f"{self.cache}/{self.source.name}/"

    def make_database(self, classes_info, database_path):
        # create df from all populations
        prediction_file_path = self.directory + f"/predictions.csv"
        headers = ["Id", "yhat", "class_no"]
        init_log(prediction_file_path, headers)

        population_file_path = self.directory + f"/populations.csv"
        df = pd.read_csv(population_file_path)
        for index, x in enumerate(df.values):
            if log_assertion(x):
                consistent = x[3]
                data = literal_eval(x[1])
                label = 1 if consistent else 0
                data.append(label)
                log_step(database_path, data)

                yhat = x[-2]
                class_no = map_float_to_class(yhat, classes_info)
                row = [index, yhat, class_no]
                log_step(prediction_file_path, row)

    def log_executed_tests(self, directory, round):
        source_executed_info_path = f"{directory}/source_executed_info.csv"
        follow_ups_executed_info_path = f"{directory}/follow_ups_executed_info.csv"
        source_info = [self.source.image_path, self.source.label, self.source.pred, self.source.label_class_no,
                       self.source.pred_class_no]
        log_step(source_executed_info_path, source_info)
        for follow_up in self.follow_ups:
            follow_up_info = [follow_up.image_path, follow_up.label, follow_up.pred, follow_up.label_class_no,
                              follow_up.pred_class_no]
            row = [round, source_info[0]]
            row.extend(follow_up_info)
            log_step(follow_ups_executed_info_path, row)

    def save_image_fearures(self, directory, round):
        self.source.save_image_features(directory, round)
        for follow_up in self.follow_ups:
            follow_up.save_image_features(directory, round)
            copy_file(follow_up.image_path, f"{directory}/executed_images/center/{round}_{follow_up.name}.jpg")

    def get_size(self):
        self.size = len(self.follow_ups) + 1
        return self.size

    def extract_follow_ups_points(self):
        for test_case in self.follow_ups:
            image_path = f"{self.directory}/{test_case.name}.jpg"
            test_case.image_path = image_path
            test_case.extract_point()

class config:
    def __init__(self, model, model_name, cache_path, label_file, parameters, device):
        self.prediction_cache_path = cache_path
        self.default_label_file = label_file
        self.parameters = parameters
        self.device = device
        self.model = model
        self.model_name = model_name

class Test_Case:
    def __init__(self, image_path=None):
        self.features = None
        self.image_path = image_path
        self.original_layer_paths = None
        self.critical_path = None
        self.critical_layer_paths = None
        self.layer_integrad_scores = None
        self.remove_threshold = None
        self.image_data = None
        # self.tensor_data = None
        self.pred = None
        self.label = None
        self.pred_class_no = None
        self.label_class_no = None
        self.point = None
        self.MR = None
        self.name = None
        self.consistent = True
        self.image_features_path = None

    def extract_pathway(self, config, model_sparsity_threshold=85):
        tensor_data = config.model.get_tensor(config.prediction_cache_path,
                                                   config.default_label_file,
                                                   config.parameters,
                                                   config.device)
        pruner = Pruner(config.model, tensor_data,
                        config.device)  # , label=self.label, output_orig=self.pred)
        pruner.prune_integrad(model_sparsity_threshold, debug=False)
        del tensor_data

        path = pruner.pruned_activations_mask
        original_layer_paths = pruner.activations
        integrad_scores = pruner.integrad_scores
        remove_threshold = pruner.remove_threshold
        pruner.remove_handles()

        layer_paths = []
        for i in range(len(path)):
            layer_paths.append(path[i].clone().cpu().detach().numpy().reshape(-1))

        path = np.hstack([layer_paths[i] for i in range(len(layer_paths))])

        layer_integrad_scores = []
        for i in range(len(integrad_scores)):
            layer_integrad_scores.append(integrad_scores[i].clone().cpu().detach().numpy().reshape(-1))

        self.critical_path = path
        self.critical_layer_paths = layer_paths
        self.original_layer_paths = original_layer_paths
        self.layer_integrad_scores = layer_integrad_scores
        self.remove_threshold = remove_threshold

    def set_pred(self, config):
        pred = config.model.predict_by_image_path(config.prediction_cache_path,
                                                       config.default_label_file, config.parameters,
                                                       config.device)
        self.pred = np.float64(pred.item())

    def calculate_distances(self, input_space):
        source_point = np.array([self.point], dtype=np.float64)
        distance = calculate_euclidean_distance(source_point, input_space)
        distance = np.array(distance, dtype=np.float64)
        mean_distance = np.mean(distance, axis=1, keepdims=True)[0][0]
        return mean_distance

    def extract_point(self):
        if isinstance(self.pred_class_no, str) or isinstance(self.label_class_no, str):
            print("ok")
        aug_features = np.array([self.pred_class_no, self.label_class_no], dtype=int)
        self.point = ExtractImageFeature(self.image_path, aug_features=aug_features, normalization=False)

    def save_image_features(self, directory, round):
        self.extract_point()
        image_feature_path = f"{directory}/executed_images_features/{round}_{self.name}.npy"
        save_npy(image_feature_path, self.point)

class MyProblem(Problem):
    def __init__(self, objectives, source_test_case, log_path, transformations, model, classes_info, config, weights=[], **kwargs):
        self.n_transformations = len(transformations) - 1
        variables = dict()

        variables[f"x01"] = Integer(bounds=(0, self.n_transformations))
        variables[f"x02"] = Real(bounds=(0.0, 1.0))

        super().__init__(vars=variables, n_obj=len(objectives), **kwargs)
        # super().__init__(
        #     n_var=n_var,
        #     n_obj=len(objectives),
        #     xl=xl,
        #     xu=xu
        # )
        self.objectives = objectives
        self.source_test_case = source_test_case
        # self.source_test_case.pred = self.get_pred()
        self.weights = weights
        self.log_path = log_path
        self.transformations = transformations
        self.model = model
        self.n_iter = 1
        self.classes_info = classes_info
        self.config = config
        headres = ["n_iter", "features", "valid_transformations",
                   "consistent",
                   "raw_objectives", "yhat",
                   "objectives"]
        init_log(self.log_path, headres)





    def _calc_pareto_front(self, n_pareto_points=1):
        return np.array([1.0] * self.n_obj)

    def test_runner(self, features, save_path=None):
        # print("--running the test...")
        new_img = copy.deepcopy(self.source_test_case.image_data)
        new_img = cv2.resize(new_img, (1280, 720))
        follow_up_name = ""
        try:
            transform_id = int(features[0])
            param = features[1]
            selected_transform = self.transformations[transform_id]
            selected_param = selected_transform[2](
                param * (selected_transform[1][1] - selected_transform[1][0]) +
                selected_transform[1][0])
            new_img, mr = selected_transform[0](new_img, selected_param)
        except Exception as e:
            print(e)
            exit(0)

        # new_img = cv2.resize(new_img[0], self.source_test_case.image_data.shape[:2])

        follow_up_test_case = Test_Case()
        follow_up_test_case.MR = mr
        follow_up_test_case.image_data = new_img

        image = follow_up_test_case.image_data

        # temporary save the image
        image_path = self.config.prediction_cache_path + "1.jpg"
        cv2.imwrite(image_path, image)

        follow_up_test_case.label = self.source_test_case.label
        follow_up_test_case.set_pred(self.config)
        follow_up_test_case.pred_class_no = map_float_to_class(follow_up_test_case.pred, self.classes_info)

        # print("--plot composed")
        # image = follow_up_test_case.image_data
        # cv2.imshow("follow up test case", cv2.resize(image, dsize=None, fx=0.3, fy=0.2))
        if save_path is not None:
            copy_file(image_path, save_path)
            # print("follow_up image saved")
        # cv2.waitKey(1)
        # plt.close()

        # print(f"--applied transformation: {follow_up_name}")
        return follow_up_test_case


    def _evaluate(self, x, out, *args, **kwargs):
        X_objectives = []
        rows = []
        for _features in x:
            features = list(_features.values())
            # print("--calculating objectives...")
            follow_up_test_case = self.test_runner(copy.deepcopy(features))

            if follow_up_test_case is None:
                x_objectives = [10 ** 9 for f in self.objectives]
                row = [self.n_iter, list(features), "", False,
                       False, True, False, x_objectives, -1]
            else:
                follow_up_test_case.features = features
                x_objectives = [f(self.source_test_case, follow_up_test_case, self.config) for f in self.objectives]
                # x_features = follow_up_test_case.features

                row = [self.n_iter, list(features), follow_up_test_case.MR, follow_up_test_case.consistent,
                       x_objectives, follow_up_test_case.pred]
            rows.append(row)
            X_objectives.append(x_objectives)

        F = np.array(X_objectives, dtype=np.float64)
        approx_ideal = F.min(axis=0)
        approx_nadir = F.max(axis=0)

        nF = []
        # normalization
        for i in range(len(x)):
            objective_list = [0] * self.n_obj
            for j in range(self.n_obj):
                divisor = (approx_nadir[j] - approx_ideal[j])
                if divisor != 0:
                    objective_list[j] = (F[i][j] - approx_ideal[j]) / divisor
            nF.append(objective_list)

        nF = 1 - np.array(nF, dtype=np.float64)
        out["F"] = nF  # minimization

        # print("--saving population...")
        for index, row in enumerate(rows):
            row.append(list(nF[index]))
            log_step(self.log_path, row)

        self.n_iter += 1

        # x = np.array(new_x)

        # out["G"] = np.array(list(zip(*X_objectives))[1])

    def rl_evaluate(self, _features):
        features = copy.deepcopy(_features)
        # print("--calculating objectives...")
        follow_up_test_case = self.test_runner(features)

        if follow_up_test_case is None:
            x_objectives = [10 ** 9 for f in self.objectives]
            # print(x_objectives)
            row = [self.n_iter, list(features), "", False,
                   False, True, False, x_objectives, -1]
        else:
            follow_up_test_case.features = features
            x_objectives = [f(self.source_test_case, follow_up_test_case, self.config) for f in self.objectives]
            # print(x_objectives)
            row = [self.n_iter, list(features), follow_up_test_case.MR, follow_up_test_case.consistent,
                   x_objectives, follow_up_test_case.pred]

        # TODO
        fitness = sum(obj * w for obj, w in zip(x_objectives, self.weights)) # maximization
        diversity = x_objectives[0]
        mr_violation = x_objectives[1]
        consistent = mr_violation != 1
        # print(f"fitness_1: {fitness_1}      fitness_2: {fitness_2}")
        # fitness = -(fitness_1 + fitness_2) # minimization
        row.append(fitness)
        log_step(self.log_path, row)
        # print(row)
        self.n_iter += 1
        modified_image = follow_up_test_case.image_data
        return mr_violation, diversity, consistent, modified_image

class ObjectiveSpaceAnimation(Callback):
    def _update(self, algorithm):
        if algorithm.n_gen % 1 == 0:
            F = algorithm.opt.get("F")
            # pf = algorithm.problem.pareto_front()

            plt.clf()
            plt.scatter(F[:, 0], F[:, 1], color="black", alpha=0.7)
            # if pf is not None:
            #     plt.plot(pf[:, 0], pf[:, 1], color="black", alpha=0.7)

            plt.show()


def brighten(img, params):
    # print("--image_brighten")
    new_img = am.brighten([img], brightness_coeff=params)[0]
    return new_img, "brighten"


def darken(img, params):
    def brightness(c: int) -> float:
        """
        Fundamental Transformation/Operation that'll be performed on
        every bit.
        """
        return 128 + params + (c - 128)

    # print("--image_darken")
    pil_img = Image.fromarray(np.uint8(img)).convert('RGB')
    new_img = pil_img.point(brightness)
    new_img = np.array(new_img)
    return new_img, "darken"


def add_snow(img, params):
    # print("--image_add_snow")
    new_img = am.add_snow([img], snow_coeff=params)[0]
    return new_img, "add_snow"


def add_rain_drizzle(img, params):
    # print("--image_add_rain_drizzle")
    new_img = am.add_rain([img], rain_type='drizzle', slant=params)[0]
    return new_img, "add_rain_drizzle"


def add_rain_heavy(img, params):
    # print("--image_add_rain_heavy")
    new_img = am.add_rain([img], rain_type='heavy', slant=params)[0]
    return new_img, "add_rain_heavy"


def add_rain_torrential(img, params):
    # print("--image_add_rain_torrential")
    new_img = am.add_rain([img], rain_type='torrential', slant=params)[0]
    return new_img, "add_rain_torrential"


def add_fog(img, params):
    # print("--image_add_fog")
    # print(params)
    new_img = am.add_fog([img], fog_coeff=params)[0]
    return new_img, "add_fog"


def add_sun_flare(img, params):
    # print("--image_add_sun_flare")
    x = y = 0
    if params == '1':
        x = random.randint(0, 100)
        y = random.randint(0, 100)
    elif params == '2':
        x = random.randint(101, 200)
        y = random.randint(101, 200)
    elif params == '3':
        x = random.randint(201, 300)
        y = random.randint(201, 300)
    elif params == '4':
        x = random.randint(301, 400)
        y = random.randint(301, 400)
    elif params == '5':
        x = random.randint(401, 500)
        y = random.randint(401, 500)
    elif params == '6':
        x = random.randint(501, 600)
        y = random.randint(501, 600)

    # print(x, y)
    new_img = am.add_sun_flare([img], flare_center=(x, y))[0]  # method call to sun_flare transformation

    return new_img, "add_sun_flare"


def add_rotation(img, params):
    # print("--add_rotation")
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), params, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst, "add_rotation"


def add_contrast(img, params):
    """
        Adjusts contrast and brightness of an uint8 image.
        contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
        brightness: [-255, 255] with 0 leaving the brightness as is
        """
    # print("--add_contrast")
    # version 1
    # new_img = cv2.multiply(img, np.array([params]))

    # version 2
    brightness = 0
    brightness += int(round(255 * (1 - params) / 2))
    new_img = cv2.addWeighted(img, params, img, 0, brightness)

    # version 3
    # new_img = cv2.convertScaleAbs(image, alpha=params, beta=0)

    return new_img, "add_contrast"


def add_translation_x(img, params):
    # print("--add_translation_x")
    if isinstance(params, list):
        params = params[0]

    rows, cols, _ = img.shape

    M = np.float32([[1, 0, params],
                    [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst, "add_translation_x"


def add_translation_y(img, params):
    # print("--add_translation_y")
    if isinstance(params, list):
        params = params[0]

    rows, cols, _ = img.shape

    M = np.float32([[1, 0, 0],
                    [0, 1, params]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst, "add_translation_y"


def add_scale(img, params):
    # print("--add_scale")
    if not isinstance(params, list):
        params = [params, params]
    res = cv2.resize(img, None, fx=params[0], fy=params[1], interpolation=cv2.INTER_LANCZOS4)
    return res, "add_scale"


def add_brightness_rgb(img, s):
    """
    Changes the image brightness by multiplying all RGB values by the same scalacar in [s_low, s_high).
    Returns the brightness adjusted image in RGB format.
    """
    # print("--add_brightness_rgb")
    img = img.astype(np.float32)
    img[:, :, :] *= s
    np.clip(img, 0, 255)
    return img.astype(np.uint8), "add_brightness_rgb"


def add_gravel(img, params):
    # print("--add_gravel")
    x1 = x2 = 0
    if params == 1:
        x1 = 0
        x2 = img.shape[1]
    elif params == 2:
        x1 = img.shape[1] * 0.2
        x2 = img.shape[1] * 0.4
    elif params == 3:
        x1 = img.shape[1] * 0.41
        x2 = img.shape[1] * 0.6
    elif params == 4:
        x1 = img.shape[1] * 0.61
        x2 = img.shape[1] * 0.8
    elif params == 5:
        x1 = img.shape[1] * 0.81
        x2 = img.shape[1]

    x1 = int(x1)
    x2 = int(x2)
    y1 = int(img.shape[0] * 0.75)
    y2 = img.shape[0]

    # print(x1, x2, y1, y2)
    new_img = am.add_gravel([img], rectangular_roi=(x1, y1, x2, y2), no_of_patches=4)[0]
    return new_img, "add_gravel"


def add_shadow(img, params):
    """
    Overlays supplied image with a random shadow polygon
    The weight range (i.e. darkness) of the shadow can be configured via the interval [w_low, w_high)
    """
    # print("--add_shadow")
    x1 = x2 = 0
    if params == 1:
        x1 = 0
        x2 = img.shape[1]
    elif params == 2:
        x1 = img.shape[1] * 0.2
        x2 = img.shape[1] * 0.4
    elif params == 3:
        x1 = img.shape[1] * 0.41
        x2 = img.shape[1] * 0.6
    elif params == 4:
        x1 = img.shape[1] * 0.61
        x2 = img.shape[1] * 0.8
    elif params == 5:
        x1 = img.shape[1] * 0.81
        x2 = img.shape[1]

    x1 = int(x1)
    x2 = int(x2)
    y1 = int(img.shape[0] * 0.75)
    y2 = img.shape[0]

    # print(x1, x2, y1, y2)

    new_img = am.add_shadow(img, no_of_shadows=1, rectangular_roi=(x1, y1, x2, y2), shadow_dimension=4)
    return new_img, "add_shadow"


def add_blur(img, params):
    # print("--add_blur")
    new_img = None
    if params == 1:
        new_img = cv2.blur(img, (3, 3))
    elif params == 2:
        new_img = cv2.blur(img, (4, 4))
    elif params == 3:
        new_img = cv2.blur(img, (5, 5))
    elif params == 4:
        new_img = cv2.GaussianBlur(img, (3, 3), 0)
    elif params == 5:
        new_img = cv2.GaussianBlur(img, (5, 5), 0)
    elif params == 6:
        new_img = cv2.GaussianBlur(img, (7, 7), 0)
    elif params == 7:
        new_img = cv2.medianBlur(img, 3)
    elif params == 8:
        new_img = cv2.medianBlur(img, 5)
    elif params == 9:
        new_img = cv2.blur(img, (6, 6))
    elif params == 10:
        new_img = cv2.bilateralFilter(img, 9, 75, 75)
    return new_img, "add_blur"


def add_speed(img, params):
    # print("--image_add_speed")
    # print(params)
    new_img = am.add_speed([img], speed_coeff=params)[0]

    return new_img, "add_speed"


def init_log(log_path, headres):
    if os.path.exists(log_path):
        return
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headres)


def log_step(log_path, row):
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def log_assertion(x):
    if x[2] is None or x[2] == "":  # bad images
        return False
    if not (isinstance(x[-1], float)):
        if "nan" in x[-1]:
            return False
    elif "nan" == x[-1]:
        return False


    dist = literal_eval(x[-3])[0]  # raw objective
    if dist == 10 ** 9:  # bad images
        return False
    return True


def mo_heuristic_mutant_generator(search_algorithm, model, model_name, test_suite, classes_info, database_path,
                                  inconsistent_pool_path,model_parameters, prediction_cache_path, default_label_file,
                                  device, inconsistents_NBC=None, time_budget="00:10:00", gama=0.2):
    relative_dirctory = test_suite.directory
    fronts_csv_file = f"{relative_dirctory}/fronts.csv"
    follow_ups_csv_file = f"{relative_dirctory}/follow_ups.csv"
    fronts_image = f"{relative_dirctory}/fronts.jpg"
    HV_image = f"{relative_dirctory}/HV"
    pathlib.Path(relative_dirctory).mkdir(parents=True, exist_ok=True)
    log_path = f"{relative_dirctory}/populations.csv"
    history_path = f"{relative_dirctory}/history.obj"
    cache_inconsistent_path = f"{relative_dirctory}/inconsistents"


    # print("--setup mutant generator...")
    source_test_case = test_suite.source


    crossover = {
        Real: SBX(prob=1, eta=15),
        Integer: SBX(prob=1, eta=15, vtype=float, repair=RoundingRepair())
    }
    mutation = {
        Real: PM(prob=0.5, eta=20),
        Integer: PM(prob=0.5, eta=20, vtype=float, repair=RoundingRepair())
    }
    # selection = TournamentSelection(func_comp=comp_by_cv_then_random)
    mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination(), crossover=crossover,
                                       mutation=mutation)
    if search_algorithm == "NSGAII":
        algorithm = NSGA2(
            pop_size=10,
            sampling=MixedVariableSampling(),
            mating=mating,
           eliminate_duplicates=MixedVariableDuplicateElimination()
        )
    elif search_algorithm == "DNSGAII":
        algorithm = DNSGA2(
            pop_size=10,
            sampling=MixedVariableSampling(),
            mating=mating,
            eliminate_duplicates=MixedVariableDuplicateElimination()
        )
    else:
        algorithm = multi_RandomSearch(
            n_points_per_iteration=10,
            sampling=MixedVariableSampling(),
            opt=None
        )

    config_obj = config(model, model_name, prediction_cache_path, default_label_file, model_parameters, device)
    source_test_case.extract_pathway(config_obj)
    max_n_clauses = sum(test_suite.source.critical_path)
    objectives = Objectives(inconsistent_pool_path, cache_inconsistent_path, max_n_clauses, inconsistents_NBC, gama)
    mo_problem = MyProblem([objectives.f1, objectives.f2], source_test_case, log_path,
                           transformations, model, classes_info, config_obj)


    running_metric = RunningMetricAnimation(delta_gen=10,
                                            tol=0.0025,
                                            key_press=True,
                                            do_show=False,
                                            indicator="hv",
                                            save_path=test_suite.directory)
    class CostimizedMultiObjectiveTermination(DefaultMultiObjectiveTermination):

        def __init__(self, n_max_evals=100000, max_time="01:00:00") -> None:
            super().__init__()

            self.max_evals = MaximumFunctionCallTermination(n_max_evals)
            self.max_time = TimeBasedTermination(max_time)

            self.criteria = [self.max_evals, self.max_time]

        def _update(self, algorithm):
            p = [criterion.update(algorithm) for criterion in self.criteria]
            return max(p)

    termination = CostimizedMultiObjectiveTermination(
        n_max_evals=300,
        max_time=time_budget
    )

    res = minimize(
        mo_problem,
        algorithm,
        termination=termination,
        callback=running_metric,
        save_history=True,
        verbose=True
    )
    if plt.get_fignums():
        plt.close()

    X, F = res.opt.get("X", "F")

    hist = res.history

    # print("saving history...")
    with open(history_path, 'wb') as filehandler:
        pickle.dump(hist, filehandler)

    print(f"-- number of pareto fronts is {len(F)}")

    population_df = pd.read_csv(log_path)
    headres = ["features", "valid_transformations", "consistent",
               "raw_objectives", "yhat", "objectives"]
    init_log(fronts_csv_file, headres)
    for _features, objectives in zip(X, F):
        features = list(_features.values())
        row = population_df.loc[(population_df["features"] == str(list(features))) & (
                population_df["objectives"] == str(list(objectives)))].values[0][1:]
        log_step(fronts_csv_file, row)

    plot = Scatter()
    plot.add(res.F, facecolor="none", edgecolor="red")
    plot.save(fronts_image)

    n_evals = []  # corresponding number of function evaluations\
    hist_F = []  # the objective space values in each generation
    hist_cv = []  # constraint violation in each generation
    hist_cv_avg = []  # average constraint violation in the whole population

    for algo in hist:
        # store the number of function evaluations
        n_evals.append(algo.evaluator.n_eval)

        # retrieve the optimum from the algorithm
        opt = algo.opt

        # store the least contraint violation and the average in each population
        hist_cv.append(opt.get("CV").min())
        hist_cv_avg.append(algo.pop.get("CV").mean())

        # filter out only the feasible and append and objective space values
        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])

    metric = Hypervolume(ref_point=np.array([1, 1], dtype=float))

    hv = [metric.do(_F) for _F in hist_F]

    with open(f"{test_suite.directory}/metric_generation_post.txt", "w") as txt_file:
        for line in hv:
            txt_file.write(str(line) + "\n")

    figure = plt.figure(figsize=(7, 5))
    plt.plot(n_evals, hv, color='black', lw=0.7, label="Avg. CV of Pop")
    plt.scatter(n_evals, hv, facecolor="none", edgecolor='black', marker="p")
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    figure.tight_layout()
    plt.savefig(HV_image + f"_{hv[-1]}.jpg")
    plt.close()

    rows = pd.read_csv(fronts_csv_file).values
    init_log(follow_ups_csv_file, headres)

    follow_ups = []
    for index, row in enumerate(rows):
        name = f"{source_test_case.name}_follow_up_{index}"
        image_path = f"{test_suite.directory}/{name}.jpg"
        features = literal_eval(row[0])
        log_step(follow_ups_csv_file, row)

        follow_up = mo_problem.test_runner(features, image_path)
        follow_up.image_path = image_path
        follow_up.name = name
        follow_up.label_class_no = test_suite.source.label_class_no
        follow_up.pred_class_no = map_float_to_class(follow_up.pred, classes_info)
        follow_ups.append(follow_up)

    test_suite.follow_ups = follow_ups
    test_suite.make_database(classes_info, database_path)


def random_mutant_generator(test_suite, classes_info, database_path, model, model_name, inconsistent_pool_path,
                            model_parameters, prediction_cache_path, default_label_file, device, gama):
    # n_var = 2
    # xl = np.array([0, 0], dtype=float)
    # xu = np.array([1, 1], dtype=float)
    # n_mutants = 100
    relative_dirctory = f"{test_suite.cache}/{test_suite.source.name}"
    pathlib.Path(relative_dirctory).mkdir(parents=True, exist_ok=True)
    log_path = f"{relative_dirctory}/populations.csv"
    cache_inconsistent_path = f"{relative_dirctory}/inconsistents"


    config_obj = config(model, model_name, prediction_cache_path, default_label_file, model_parameters, device)
    objectives = Objectives(inconsistent_pool_path, save_path=cache_inconsistent_path, gama=gama)
    simple_problem = MyProblem([objectives.f2], test_suite.source, log_path, transformations, model, classes_info, config_obj)

    algorithm = RandomSearch(
        n_points_per_iteration=10,
        sampling=MixedVariableSampling()
    )
    algorithm.problem = simple_problem
    pop = algorithm._infill()
    x = np.array([individual.x for individual in pop])
    out = {"F": np.array([])}
    simple_problem._evaluate(x, out)

    test_suite.make_database(classes_info, database_path)


def run_mutant_generator(search_algorithm, *args, **kwargs):
    output = None
    mo_heuristic_mutant_generator(search_algorithm, *args, **kwargs)
    return output
    

transformations = [
    # weather transformations: https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    (add_sun_flare, (1, 6), int),
    (add_blur, (1, 10), int),
    (add_snow, (0, 1), float),
    (add_rain_torrential, (-10, 10), int),
    (add_shadow, (1, 5), int),
    (add_translation_y, (-50, 49), int),
    (add_fog, (0.1, 1), float),
    (add_translation_x, (-50, 49), int),
    (add_speed, (0, 1), float),
    (add_gravel, (1, 5), int),
    (darken, (-50, -1), int),
    (add_rain_heavy, (-10, 10), int),
    (add_brightness_rgb, (0.2, 0.75), float),
    (add_rotation, (5, 10), int),
    (add_scale, (0.5, 1.9), float),
    (add_rain_drizzle, (-10, 10), int),
    (add_contrast, (0.5, 1.9), float),
    (brighten, (0, 1), float)
]

if __name__ == "__main__":
    pass

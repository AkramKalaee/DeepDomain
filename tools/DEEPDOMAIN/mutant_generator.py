import sys
import os

from constants import SRC_DIR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import csv
import functools
import glob
import math
import pickle
import pandas as pd
import time
import cv2
from tools.DEEPDOMAIN.termination.running_metric import RunningMetricAnimation
from tools.DEEPDOMAIN.pathway_grad.src import Pruner, Plot_tools
import pathlib
import matplotlib.pyplot as plt
import torch
import imageio
import tools.DEEPDOMAIN.automold.Automold as am
import tools.DEEPDOMAIN.automold.Helpers as hp
from PIL import Image
from torchvision import datasets, models, transforms
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split
import glob
import itertools
import cv2
import faiss
import imageio
import numpy as np
import tensorflow as tf
from tools.DEEPDOMAIN.pathway_grad.src.Pruner import Pruner
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.layers import (Convolution2D, Dense, Dropout, Flatten, Input,
                                     Lambda, MaxPooling2D)
from tensorflow.keras.models import Model
import cv2
import pickle
import copy
import time
import os
import pickle
import copy
import random
import pickle
import math
import cv2
import random
from sklearn.metrics import mean_squared_error
import pickle
import glob
import numpy as np
import random
import sys
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class RS:
    def __init__(self, test_budget, generate_individual,  calculate_objectives, fast_nondominated_sort):
        self.test_budget = test_budget
        self.fast_nondominated_sort = fast_nondominated_sort
        self.generate_individual = generate_individual
        self.calculate_objectives = calculate_objectives

    def run(self):
        population = Population()
        start_time = time.time()
        while True:
            if time.time() - start_time > self.test_budget:
                break
            individual = self.generate_individual()
            self.calculate_objectives(individual)
            population.append(individual)

        print("--fast nondominated sorting...")
        self.fast_nondominated_sort(population)
        print("done")
        return population.fronts


class Test_Data:
    def __init__(self, image_path=None):

        self.image_path = image_path
        self.original_layer_paths = None
        self.critical_path = None
        self.critical_layer_paths = None
        self.layer_integrad_scores = None
        self.remove_threshold = None
        self.image_data = None
        self.tensor_data = None
        self.pred = None
        self.original_pred = None
        self.name = None
        self.feasible = False
        self.delta_boundary = False
        self.consistent = True
        self.v_unsafe = False

    def set_pred(self, model):
        model.eval()
        with torch.no_grad():
            prediction = model(self.tensor_data)
        # self.pred = prediction.detach().cpu().flatten().numpy().tolist() #out.item()
        self.pred = np.float64(prediction.item())

    def prepare_data(self, target_size=(120, 320)):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
        if self.image_path is not None:
            frame = cv2.imread(self.image_path)
            orig_frame = frame.copy()
            self.image_data = orig_frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = self.image_data.copy()

        frame = frame[65:-25, :, :]
        frame = transform(frame)
        frame = frame.unsqueeze(0).to(device)
        self.tensor_data = frame

    def extract_pathway(self, model, model_sparsity_threshold=95):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pruner = Pruner(model, self.tensor_data, device)
        pruner.prune_integrad(model_sparsity_threshold, debug=False)
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

# nsgaii.py
class Evolution:

    def __init__(self, problem, utils, test_budget, save_path, algorithm):
        self.utils = utils
        self.population = None
        self.on_generation_finished = []
        self.num_of_individuals = utils.num_of_individuals
        self.problem = problem
        self.test_budget = test_budget
        self.save_path = save_path
        self.algorithm = algorithm

    def register_on_new_generation(self, fun):
        self.on_generation_finished.append(fun)

    def evolve(self):
        last_solution = None
        start_time = time.time()
        print("--creating initial population...")
        self.population = self.utils.create_initial_population()

        print("--fast nondominated sorting...")
        self.utils.fast_nondominated_sort(self.population)
        print("--calculating crowding distance")
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
        print("--creating children")
        children = self.utils.create_children(self.population)
        print("--evoulution starting...")

        returned_population = None
        generation_counter = 1
        while True:
            if time.time() - start_time > self.test_budget:
                break
            self.problem.generation_index = generation_counter

            self.population.extend(children)

            print("--fast nondominated sorting...")
            self.utils.fast_nondominated_sort(self.population)

            print("--calculating crowding distance")
            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)

            new_population = Population()
            front_num = 0

            print("--collecting new population...")
            while len(new_population) + len(self.population.fronts[front_num]) <= self.num_of_individuals:
                new_population.extend(self.population.fronts[front_num])

                if len(self.population.fronts) <= front_num + 1:
                    break
                else:
                    front_num += 1

            if self.num_of_individuals > len(new_population):
                self.population.fronts[front_num] = sorted(self.population.fronts[front_num], key=functools.cmp_to_key(self.utils.crowding_operator))
                new_population.extend(
                    self.population.fronts[front_num][0:self.num_of_individuals - len(new_population)])

            returned_population = self.population
            self.population = new_population

            print("--creating children")
            children = self.utils.create_children(self.population)

            print("--calculating metrics...")
            self.algorithm.population = returned_population
            self.algorithm.n_gen = generation_counter
            self.algorithm.n_evals = self.problem.n_evals
            for fun in self.on_generation_finished:
                fun(self.algorithm, self.utils)

            print("--saving population...")
            self.population.save(self.save_path, generation_counter)

            if self.algorithm.force_termination:
                print("--evolution converged")
                break

            generation_counter += 1

        print("--evolution completed")
        return returned_population.fronts


class Algorithm:
    def __init__(self, save_path):
        self.force_termination = False
        self.F = None
        self.n_gen = None
        self.n_evals = None
        self.save_path = save_path
        self.population = None

class Metrics(RunningMetricAnimation):
    def __init__(self):
        super().__init__(delta_gen=10,
                         n_plots=3,
                         key_press=False,
                         do_show=True)

    def convergence(self, algorithm, utils):
        print("--check convergence")
        # version 1
        # metric_populaton = Population()
        # for individual in algorithm.population:
        #     cost = len([t for t in self.transformations_list if t in individual.name]) #number of transformations
        #     cost = 10**9 if cost == 0 else cost
        #     effectiveness = copy.copy(individual.objectives[1])
        #     metric_individual = Individual()
        #     metric_individual.objectives = [cost, effectiveness]
        #     metric_populaton.append(metric_individual)
        #
        # #calculate fronts
        # utils.fast_nondominated_sort(metric_populaton)
        # F = []
        # for individual in metric_populaton.fronts[0]:
        #     F.append(individual.objectives)

        # version 2
        # F = []
        # for individual in algorithm.population.fronts[0]:
        #     F.append(individual.objectives)

        # version 3
        # send only solution set
        F = []
        for individual in algorithm.population.fronts[0]:
            F.append(individual.objectives)

        algorithm.F = np.array(F)
        # self.do(None, algorithm, force_plot=False)

        return algorithm.force_termination

    def record_hisotry(self, algorithm, utils):
        print("--recording history")
        log_path = f"{algorithm.save_path}/history.csv"
        headers = ["n_gen", "n_evals", "F", "ideal", "nadir"]

        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        current = self.running.history[-1]
        row = [algorithm.n_gen, algorithm.n_evals, current["F"],current["ideal"],current["nadir"]]
        log_step(log_path, row)

    def objective_space_animation(self, algorithm, utils):
        if algorithm.n_gen % 1 == 0:
            F = algorithm.F
            # pf = algorithm.problem.pareto_front()

            plt.clf()
            plt.scatter(F[:, 0], F[:, 1], color="black", alpha=0.7)
            # if pf is not None:
            #     plt.plot(pf[:, 0], pf[:, 1], color="black", alpha=0.7)

            plt.show()
# population.py
class Population:
    def __init__(self):
        self.population = []
        self.fronts = []

    def __len__(self):
        return len(self.population)

    def __iter__(self):
        return self.population.__iter__()

    def extend(self, new_individuals):
        self.population.extend(new_individuals)

    def append(self, new_individual):
        self.population.append(new_individual)

    def save(self, path, name):
        with open(f"{path}/generation_{name}.obj", 'wb') as filehandler:
            pickle.dump(self, filehandler)

# individual.py
class Individual(Test_Data):
    def __init__(self):
        super().__init__()
        self.rank = None
        self.crowding_distance = None
        self.domination_count = None
        self.dominated_solutions = None
        self.features = None
        self.features_choice = None
        self.objectives = None

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.features == other.features

        return False

    def dominates(self, other_individual):
        and_condition = True
        or_condition = False
        for first, second in zip(self.objectives, other_individual.objectives):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second

        return (and_condition and or_condition)

    def save(self, filepath):
        with open(filepath, 'wb') as filehandler:
            pickle.dump(self, filehandler)

# problem.py
class Problem:
    def __init__(self, objectives, num_of_variables, variables_range, transformations,
                 validation, source_test_data, log_path, model, same_range=False):
        self.num_of_objectives = len(objectives)
        self.num_of_variables = num_of_variables
        self.objectives = objectives
        self.transformations = transformations
        self.variables_range = []
        if same_range:
            for _ in range(num_of_variables):
                self.variables_range.append(variables_range[0])
        else:
            self.variables_range = variables_range

        self.validation = validation
        self.source_test_data = source_test_data
        self.log_path = log_path
        self.model = model
        self.n_evals = 0
        self.generation_index = 0
        headres = ["generation_no", "features", "valid_transformations", "feasible", "delta_boundary", "consistent", "v_unsafe",
                   "objectives"]
        init_log(self.log_path, headres)

    def test_runner_v1(self, individual):
        print("--running the test...")
        transforms = list(map(lambda i: (self.transformations[i][0], self.transformations[i][2](
            individual.features[i] * (self.transformations[i][1][1] - self.transformations[i][1][0]) + self.transformations[i][1][0])),
                              range(len(self.transformations))))

        print("--finding valid transformations...")
        filtered_transforms = []
        invalid_transformations_index = []
        for i, (transform, param) in enumerate(transforms):
            # try:
            flag = True
            new_img = transform(copy.deepcopy(hp.load_images(self.source_test_data.image_path)), param)
            if new_img is None:
                flag = False
            else:
                new_img = cv2.resize(new_img[0], self.source_test_data.image_data.shape[:2])
            # except:
            #     flag = False

            if flag:
                # cv2.imshow("follow_up", new_img)
                # cv2.waitKey(1)
                # plt.close()

                follow_up_test_data = Test_Data()
                follow_up_test_data.image_data = new_img
                follow_up_test_data.prepare_data()
                follow_up_test_data.set_pred(self.model)

                if self.validation(self.source_test_data, follow_up_test_data):
                    filtered_transforms.append((transform, param))
                else:
                    flag = False

            if not flag:
                invalid_transformations_index.append(i)


        new_img = copy.deepcopy(self.source_test_data.image_data)
        valid_transformations = []
        for i, (transform, param) in enumerate(filtered_transforms):
            try:
                new_img = transform(new_img, param)
                valid_transformations.append(f"{transform.__name__}_{param}")
            except:
                invalid_transformations_index.append(i)


        n_valid_transformations = len(valid_transformations)
        print(f"--#valid transformations: {n_valid_transformations}")

        if n_valid_transformations > 0:
            follow_up_name = "_".join(valid_transformations)
            individual.name = follow_up_name
            individual.image_data = new_img
            individual.prepare_data()
            individual.set_pred(self.model)
            individual.extract_pathway(self.model)


        n_invalid_transformations = len(invalid_transformations_index)
        if n_invalid_transformations > 0:
            # recorrection the gens
            for i in invalid_transformations_index:
                individual.features[i] = self.transformations[i][3]

    def test_runner_v2(self, individual):
        print("--running the test...")
        transforms = list(map(lambda i: (self.transformations[i][0], self.transformations[i][2](
            individual.features[i] * (self.transformations[i][1][1] - self.transformations[i][1][0]) + self.transformations[i][1][0])),
                              range(len(self.transformations))))


        print("--finding valid transformations...")
        unapplied_transformations_index = []
        new_img = copy.deepcopy(hp.load_images(self.source_test_data.image_path))
        valid_transformations = []
        for i, (transform, param) in enumerate(transforms):
            try:
                current_img = transform(new_img, param)
                if current_img is None:
                    unapplied_transformations_index.append(i)
                else:
                    valid_transformations.append(f"{transform.__name__}")
                    new_img = current_img
            except:
                unapplied_transformations_index.append(i)


        n_valid_transformations = len(valid_transformations)
        print(f"--#valid transformations: {n_valid_transformations}")

        if n_valid_transformations > 0:
            new_img = cv2.resize(new_img[0], self.source_test_data.image_data.shape[:2])
            follow_up_name = "_".join(valid_transformations)
            individual.name = follow_up_name
            individual.image_data = new_img
            individual.prepare_data()
            individual.set_pred(self.model)
            individual.extract_pathway(self.model)

            print("--plot composed")
            image = individual.image_data
            cv2.imshow("composed", image)
            cv2.waitKey(1)
            plt.close()

        n_unapplied_transformations = len(unapplied_transformations_index)
        if n_unapplied_transformations > 0:
            # recorrection the gens
            for i in unapplied_transformations_index:
                individual.features[i] = self.transformations[i][3]

    def test_runner(self, individual):
        print("--running the test...")
        transforms = list(map(lambda i: (self.transformations[i][0], self.transformations[i][2](
            individual.features[i] * (self.transformations[i][1][1] - self.transformations[i][1][0]) + self.transformations[i][1][0]), self.transformations[i][4]),
                              range(len(self.transformations))))


        print("--finding valid transformations...")
        unapplied_transformations_index = []
        new_img = copy.deepcopy(hp.load_images(self.source_test_data.image_path))
        valid_transformations = []
        j = -1
        # , (transform, param, is_composite)
        for i in range(len(transforms)):
            try:
                j += 1
                item = transforms[j]
                param = item[1]
                is_composite = item[2]

                if is_composite:
                    j += 1
                    next_param = individual.features[j]
                    param = [param, next_param]

                transform = item[0]
                current_img, title = transform(new_img, param)
                if current_img is None:
                    unapplied_transformations_index.append(i)
                else:
                    valid_transformations.append(title)
                    new_img = current_img
            except Exception as e:
                print(e)
                exit(0)
                unapplied_transformations_index.append(i)


        n_valid_transformations = len(valid_transformations)

        follow_up_name = ""
        if n_valid_transformations > 0:
            new_img = cv2.resize(new_img[0], self.source_test_data.image_data.shape[:2])
            follow_up_name = "_".join(valid_transformations)
            individual.name = follow_up_name
            individual.image_data = new_img
            individual.prepare_data()
            individual.set_pred(self.model)
            individual.extract_pathway(self.model)

            print("--plot composed")
            image = individual.image_data
            cv2.imshow("composed", image)
            cv2.waitKey(1)
            plt.close()

        print(f"--#valid transformations: {n_valid_transformations}--> {follow_up_name}")

        n_unapplied_transformations = len(unapplied_transformations_index)
        if n_unapplied_transformations > 0:
            # recorrection the gens
            for i in unapplied_transformations_index:
                individual.features[i] = self.transformations[i][3]

    def generate_individual(self):
        individual = Individual()
        individual.features = []
        for x in self.variables_range:
            if isinstance(x[0], list):
                random_range = random.choice(x)
                individual.features.append(random.uniform(*random_range))
            else:
                individual.features.append(random.uniform(*x))

        return individual

    def calculate_objectives(self, individual):
        self.n_evals += 1
        print("--calculating objectives...")
        self.test_runner(individual)
        if individual.name is None:
            individual.objectives = [10**9 for f in self.objectives]
        else:
            individual.objectives = [f(self.source_test_data, individual) for f in self.objectives]

        #local search

        print("--logging follow_up info...")
        row = [self.generation_index, individual.features, individual.name,	individual.feasible, individual.delta_boundary, individual.consistent, individual.v_unsafe,	list(individual.objectives)]
        log_step(self.log_path, row)

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

# util.py
class NSGA2:

    def __init__(self, problem, num_of_individuals=100,
                 num_of_tour_particips=2, tournament_prob=0.9, crossover_param=5, mutation_param=5):

        self.problem = problem
        self.num_of_individuals = num_of_individuals
        self.num_of_tour_particips = num_of_tour_particips
        self.tournament_prob = tournament_prob
        self.crossover_param = crossover_param
        self.mutation_param = mutation_param


    def create_initial_population(self):
        population = Population()
        for _ in range(self.num_of_individuals):
            individual = self.problem.generate_individual()
            self.problem.calculate_objectives(individual)
            population.append(individual)

        return population

    def populate_population(self, population):
        for _ in range(self.num_of_individuals - len(population)):
            individual = self.problem.generate_individual()
            self.problem.calculate_objectives(individual)
            population.append(individual)

    def fast_nondominated_sort(self, population, normalization=True):
        if normalization:
            X_objectives = []

            for individual in population:
                X_objectives.append(individual.objectives)

            F = np.array(X_objectives)
            approx_ideal = F.min(axis=0)
            approx_nadir = F.max(axis=0)
            # normalization
            nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

            for index, objectives in enumerate(nF):
                population.population[index].objectives = objectives

        population.fronts = [[]]
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in population:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                population.fronts[0].append(individual)
        i = 0
        while len(population.fronts[i]) > 0:
            temp = []
            for individual in population.fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)


            # if len(temp) > 0:
            #     i = i + 1
            #     population.fronts.append(temp)
            # else:
            #     break
            i = i + 1
            population.fronts.append(temp)

    def calculate_crowding_distance(self, front):
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            for m in range(len(front[0].objectives)):
                front.sort(key=lambda individual: individual.objectives[m])
                front[0].crowding_distance = 10 ** 9
                front[solutions_num - 1].crowding_distance = 10 ** 9
                m_values = [individual.objectives[m] for individual in front]
                scale = max(m_values) - min(m_values)
                if scale == 0: scale = 1
                for i in range(1, solutions_num - 1):
                    front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / scale


    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or \
                ((individual.rank == other_individual.rank) and (
                        individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def create_children(self, population):
        children = []
        while len(children) < len(population):
            parent1 = self.__tournament(population)
            parent2 = parent1
            print("tornumenting...")
            while parent1.features == parent2.features:
                parent2 = self.__tournament(population)
            print("crossover...")
            child1, child2 = self.__crossover(parent1, parent2)
            print("mutating..")
            self.__mutate(child1)
            self.__mutate(child2)
            print("calculating objectives...")
            self.problem.calculate_objectives(child1)
            self.problem.calculate_objectives(child2)
            children.append(child1)
            children.append(child2)

        return children

    def __crossover(self, individual1, individual2):
        child1 = self.problem.generate_individual()
        child2 = self.problem.generate_individual()
        num_of_features = len(child1.features)
        genes_indexes = range(num_of_features)
        for i in genes_indexes:
            beta = self.__get_beta()
            x1 = (individual1.features[i] + individual2.features[i]) / 2
            x2 = abs((individual1.features[i] - individual2.features[i]) / 2)
            child1.features[i] = x1 + beta * x2
            child2.features[i] = x1 - beta * x2
        return child1, child2

    def __get_beta(self):
        u = random.random()
        if u <= 0.5:
            return (2 * u) ** (1 / (self.crossover_param + 1))
        return (2 * (1 - u)) ** (-1 / (self.crossover_param + 1))

    def __mutate(self, child):
        num_of_features = len(child.features)
        for gene in range(num_of_features):
            # select the range that child had chosed before, while had been generated
            variable_range = self.problem.variables_range[gene][child.features_choice[gene]] if isinstance(
                self.problem.variables_range[gene][0], list) else self.problem.variables_range[gene]

            u, delta = self.__get_delta()
            if u < 0.5:
                child.features[gene] += delta * (child.features[gene] - variable_range[0])
            else:
                child.features[gene] += delta * (variable_range[1] - child.features[gene])

            if child.features[gene] < variable_range[0]:
                child.features[gene] = variable_range[0]
            elif child.features[gene] > variable_range[1]:
                child.features[gene] = variable_range[1]

    def __get_delta(self):
        u = random.random()
        if u < 0.5:
            return u, (2 * u) ** (1 / (self.mutation_param + 1)) - 1
        return u, 1 - (2 * (1 - u)) ** (1 / (self.mutation_param + 1))

    def __tournament(self, population):
        participants = random.sample(population.population, self.num_of_tour_particips)
        best = None
        for participant in participants:
            if best is None or (
                    self.crowding_operator(participant, best) == 1 and self.__choose_with_prob(self.tournament_prob)):
                best = participant

        return best

    def __choose_with_prob(self, prob):
        if random.random() <= prob:
            return True
        return False

class Objectives:
    def __init__(self):
        self.lambda_ = 5
        self.epsilon_ = 0.03
        self.K = 0.0001
        self.delta = 0.5
        self.gama = 0.2  # [0.1, 0.2, 0.3]  [10, 20, 30, 40]

    def f1(self, source_test_data, follow_up_test_data):
        """
        Structural metric
        """
        def normalize(value):
            return value / (value + 1)

        # TODO: finitto
        def calculate_approach_level(path1, path2):
            distance = sum(path1) - np.dot(path1, path2)
            return distance

        # TODO
        def calculate_branch_distance_v1(threshold, layer_path1, layer_path2, K, delta):
            distance = -1e+8
            n_layers = len(layer_path1)

            for i in range(n_layers):
                # branch: scores <= remove_threshold
                n_neurons = len(layer_path1[i])
                for j in range(n_neurons):
                    score = layer_path2[i][j]
                    if layer_path1[i][j] == 0:  # true branch: scores <= threshold
                        if score <= threshold:
                            dist = - ((threshold - score) + K)
                        else:
                            dist = score - threshold
                    else:                     # true branch: score > threshold
                        if score > threshold:
                            dist = -(score - threshold)
                        else:
                            dist = (threshold - score) + K
                    distance = max(distance, dist)


            feasible = delta_boundary = False
            # version 1
            # if distance < 0:
            #     feasible = True
            #     if distance >= -delta:
            #         delta_boundary = True
            #         distance = 0
            #     else:
            #         distance = abs(distance)

            # version 2
            if distance < 0:
                feasible = True
                if distance >= -delta:
                    delta_boundary = True
                distance = 0

            return distance, feasible, delta_boundary

        def calculate_branch_distance(threshold, layer_path1, layer_path2, K, delta):
            distance = 0
            n_layers = len(layer_path1)

            for i in range(n_layers):
                # branch: scores <= remove_threshold
                n_neurons = len(layer_path1[i])
                for j in range(n_neurons):
                    score = layer_path2[i][j]
                    if layer_path1[i][j] == 0:  # true branch: scores <= threshold
                        if score <= threshold:
                            dist = 0
                        else:
                            dist = score - threshold
                    else:  # true branch: score > threshold
                        if score > threshold:
                            dist = 0
                        else:
                            dist = (threshold - score) + K
                    distance += dist

            feasible = delta_boundary = False

            return distance, feasible, delta_boundary


        # similarity
        approach_level = calculate_approach_level(source_test_data.critical_path, follow_up_test_data.critical_path)
        branch_distance, feasible, delta_boundary = calculate_branch_distance(source_test_data.remove_threshold,
                                                                              source_test_data.critical_layer_paths,
                                                                              follow_up_test_data.layer_integrad_scores,
                                                                              self.K, self.delta)

        # cost
        # approach_level = calculate_approach_level(follow_up_test_data.critical_path, source_test_data.critical_path)
        # branch_distance, feasible, delta_boundary = calculate_branch_distance(follow_up_test_data.remove_threshold,
        #                                                                       follow_up_test_data.critical_layer_paths,
        #                                                                       source_test_data.layer_integrad_scores,
        #                                                                       self.K, self.delta)
        #
        fitness = approach_level + normalize(branch_distance)
        # fitness = -(approach_level + normalize(branch_distance))
        # fitness = abs(source_test_data.remove_threshold - follow_up_test_data.remove_threshold)
        # fitness = normalize(abs(source_test_data.remove_threshold - follow_up_test_data.remove_threshold) + approach_level)
        # fitness = abs(source_test_data.remove_threshold - follow_up_test_data.remove_threshold) + normalize(branch_distance)
        # fitness = abs(source_test_data.remove_threshold - follow_up_test_data.remove_threshold) + branch_distance
        # fitness = normalize(abs(source_test_data.remove_threshold - follow_up_test_data.remove_threshold)) + branch_distance

        print(f"src: {source_test_data.remove_threshold}        follow_up: {follow_up_test_data.remove_threshold}")  #   ap-level: {approach_level}      branch_dist: {branch_distance}     features: {follow_up_test_data.features}")

        # follow_up_test_data.feasible = feasible
        # follow_up_test_data.delta_boundary = delta_boundary

        print(f"--f1 : {fitness}") #    feasible: {feasible}     delta_boundary: {delta_boundary}")
        return fitness

    def f2_v1(self, source_test_data, follow_up_test_data):
        """
        Metamorphic rule
        """
        def normalize(value):
            return value / (value + 1)

        # ref.: deep_test
        original_label = source_test_data.label
        original_pred = source_test_data.pred
        transformed_pred = follow_up_test_data.pred
        mse_orig = mean_squared_error(y_true=[original_label], y_pred=[original_pred])
        mse_transformed = mean_squared_error(y_true=[original_label], y_pred=[transformed_pred])
        consistent = True
        if math.pow(original_label - transformed_pred, 2) <= self.lambda_ * mse_orig:
            MR1 = ((self.lambda_ * mse_orig) - math.pow(original_label - transformed_pred, 2) + self.K)
        else:
            MR1 = - (math.pow(original_label - transformed_pred, 2) - (self.lambda_ * mse_orig))

        if np.abs(mse_transformed - mse_orig) <= self.epsilon_:
            MR2 = - ((self.epsilon_ - np.abs(mse_transformed - mse_orig)) + self.K)
            consistent = False
        else:
            MR2 = (np.abs(mse_transformed - mse_orig) - self.epsilon_)

        dist = max(MR1, MR2)
        if dist < 0:
            distance = 1 - normalize(abs(dist))
        else:
            distance = normalize(dist)

        follow_up_test_data.consistent = consistent
        print(f"--f2 : {distance}    consistent: {consistent}")

        return distance

    def f2_v2(self, source_test_data, follow_up_test_data):
        """
        Metamorphic rule
        """
        def normalize(value):
            return value / (value + 1)

        # ref.: deep_test
        original_label = source_test_data.label
        original_pred = source_test_data.pred
        transformed_pred = follow_up_test_data.pred
        mse_orig = mean_squared_error(y_true=[original_label], y_pred=[original_pred])
        mse_transformed = mean_squared_error(y_true=[original_label], y_pred=[transformed_pred])
        consistent = True
        if math.pow(original_label - transformed_pred, 2) <= self.lambda_ * mse_orig:
            MR1 = ((self.lambda_ * mse_orig) - math.pow(original_label - transformed_pred, 2) + self.K)
        else:
            MR1 = - (math.pow(original_label - transformed_pred, 2) - (self.lambda_ * mse_orig))

        if np.abs(mse_transformed - mse_orig) <= self.epsilon_:
            MR2 = 0
            consistent = False
        else:
            MR2 = (np.abs(mse_transformed - mse_orig) - self.epsilon_)

        v_unsafe = False
        if MR1 < 0:
            if MR1 >= -self.delta:
                v_unsafe = True
                dist = 0
            else:
                dist = abs(MR1)
        else:
            dist = MR1

        distance = dist + normalize(MR2)

        follow_up_test_data.consistent = consistent
        follow_up_test_data.v_unsafe = v_unsafe
        print(f"--f2 : {distance}    consistent: {consistent}       v_unsafe: {v_unsafe}")

        return distance

    def f2_v3(self, source_test_data, follow_up_test_data):
        """
        Metamorphic rule
        """
        def normalize(value):
            return value / (value + 1)

        # ref.: deep_test
        original_label = source_test_data.label
        original_pred = source_test_data.pred
        transformed_pred = follow_up_test_data.pred
        mse_orig = mean_squared_error(y_true=[original_label], y_pred=[original_pred])
        consistent = True
        if math.pow(original_label - transformed_pred, 2) <= self.lambda_ * mse_orig:
            MR1 = ((self.lambda_ * mse_orig) - math.pow(original_label - transformed_pred, 2) + self.K)
        else:
            MR1 = - (math.pow(original_label - transformed_pred, 2) - (self.lambda_ * mse_orig))
            consistent = False

        v_unsafe = False
        # version 1
        # if MR1 < 0:
        #     if MR1 >= -self.delta:
        #         v_unsafe = True
        #         distance = 0
        #     else:
        #         distance = abs(MR1)
        # else:
        #     distance = MR1

        # version 2
        if MR1 < 0:
            if MR1 >= -self.delta:
                v_unsafe = True
        distance = MR1

        follow_up_test_data.consistent = consistent
        follow_up_test_data.v_unsafe = v_unsafe
        print(f"--f2 : {distance}    consistent: {consistent}       v_unsafe: {v_unsafe}")

        return distance

    def f2(self, source_test_data, follow_up_test_data):
        """
        Metamorphic rule
        """

        def normalize(value):
            return value / (value + 1)

        # ref.: deep_test
        # original_label = source_test_data.label
        original_pred = source_test_data.pred
        transformed_pred = follow_up_test_data.pred

        consistent = True
        lhs = abs(original_pred - transformed_pred)
        rhs = self.gama
        if lhs <= rhs:
            MR1 = rhs - lhs + self.K
        else:
            MR1 = - (lhs - rhs)
            consistent = False

        v_unsafe = False
        # version 1
        # if MR1 < 0:
        #     if MR1 >= -self.delta:
        #         v_unsafe = True
        #         distance = 0
        #     else:
        #         distance = abs(MR1)
        # else:
        #     distance = MR1

        # version 2
        if MR1 < 0:
            if MR1 >= -self.delta:
                v_unsafe = True
        distance = MR1

        follow_up_test_data.consistent = consistent
        follow_up_test_data.v_unsafe = v_unsafe
        print(f"--f2 : {distance}    consistent: {consistent}       v_unsafe: {v_unsafe}")

        return distance

    def f3(self, source_test_data, follow_up_test_data):
        original_label = source_test_data.label
        original_pred = source_test_data.pred
        transformed_pred = follow_up_test_data.pred
        mse_transformed = mean_squared_error(y_true=[original_label], y_pred=[transformed_pred])
        mse_orig = mean_squared_error(y_true=[original_label], y_pred=[original_pred])
        result = np.abs(mse_transformed - mse_orig) <= self.epsilon_

        print(f"--f3 : {result} ")
        return result

# class Objectives:
#     def __init__(self):
#         self.lambda_ = 5
#         self.epsilon_ = 0.03
#         self.K = 0.0001
#         self.delta = 0.5
#         self.gama = 0.2  # [0.1, 0.2, 0.3]  [10, 20, 30, 40]
#
#     def f1(self, source_test_data, follow_up_test_data):
#         """
#         Structural metric
#         """
#         def normalize(value):
#             return value / (value + 1)
#
#         # TODO: finitto
#         def calculate_approach_level(path1, path2):
#             distance = sum(path1) - np.dot(path1, path2)
#             return distance
#
#         # TODO
#         def calculate_branch_distance_v1(threshold, layer_path1, layer_path2, K, delta):
#             distance = -1e+8
#             n_layers = len(layer_path1)
#
#             for i in range(n_layers):
#                 # branch: scores <= remove_threshold
#                 n_neurons = len(layer_path1[i])
#                 for j in range(n_neurons):
#                     score = layer_path2[i][j]
#                     if layer_path1[i][j] == 0:  # true branch: scores <= threshold
#                         if score <= threshold:
#                             dist = - ((threshold - score) + K)
#                         else:
#                             dist = score - threshold
#                     else:                     # true branch: score > threshold
#                         if score > threshold:
#                             dist = -(score - threshold)
#                         else:
#                             dist = (threshold - score) + K
#                     distance = max(distance, dist)
#
#
#             feasible = delta_boundary = False
#             # version 1
#             # if distance < 0:
#             #     feasible = True
#             #     if distance >= -delta:
#             #         delta_boundary = True
#             #         distance = 0
#             #     else:
#             #         distance = abs(distance)
#
#             # version 2
#             if distance < 0:
#                 feasible = True
#                 if distance >= -delta:
#                     delta_boundary = True
#                 distance = 0
#
#             return distance, feasible, delta_boundary
#
#         def calculate_branch_distance(threshold, layer_path1, layer_path2, K, delta):
#             distance = 0
#             n_layers = len(layer_path1)
#
#             for i in range(n_layers):
#                 # branch: scores <= remove_threshold
#                 n_neurons = len(layer_path1[i])
#                 for j in range(n_neurons):
#                     score = layer_path2[i][j]
#                     if layer_path1[i][j] == 0:  # true branch: scores <= threshold
#                         if score <= threshold:
#                             dist = 0
#                         else:
#                             dist = score - threshold
#                     else:                     # true branch: score > threshold
#                         if score > threshold:
#                             dist = 0
#                         else:
#                             dist = (threshold - score) + K
#                     distance += dist
#
#             feasible = delta_boundary = False
#
#             return distance, feasible, delta_boundary
#
#         # similarity
#         approach_level = calculate_approach_level(source_test_data.critical_path, follow_up_test_data.critical_path)
#         branch_distance, feasible, delta_boundary = calculate_branch_distance(source_test_data.remove_threshold,
#                                                                               source_test_data.critical_layer_paths,
#                                                                               follow_up_test_data.layer_integrad_scores,
#                                                                               self.K, self.delta)
#
#         # cost
#         # approach_level = calculate_approach_level(follow_up_test_data.critical_path, source_test_data.critical_path)
#         # branch_distance, feasible, delta_boundary = calculate_branch_distance(follow_up_test_data.remove_threshold,
#         #                                                                       follow_up_test_data.critical_layer_paths,
#         #                                                                       source_test_data.layer_integrad_scores,
#         #                                                                       self.K, self.delta)
#
#         fitness = approach_level + normalize(branch_distance)
#         # fitness = -(approach_level + normalize(branch_distance))
#         # fitness = abs(source_test_data.remove_threshold - follow_up_test_data.remove_threshold)
#         # fitness = normalize(abs(source_test_data.remove_threshold - follow_up_test_data.remove_threshold) + approach_level)
#         # fitness = abs(source_test_data.remove_threshold - follow_up_test_data.remove_threshold) + normalize(branch_distance)
#         # fitness = abs(source_test_data.remove_threshold - follow_up_test_data.remove_threshold) + branch_distance
#         # fitness = normalize(abs(source_test_data.remove_threshold - follow_up_test_data.remove_threshold)) + branch_distance
#
#         print(f"src: {source_test_data.remove_threshold}        follow_up: {follow_up_test_data.remove_threshold}")  #   ap-level: {approach_level}      branch_dist: {branch_distance}     features: {follow_up_test_data.features}")
#
#         # follow_up_test_data.feasible = feasible
#         # follow_up_test_data.delta_boundary = delta_boundary
#
#         print(f"--f1 : {fitness}") #    feasible: {feasible}     delta_boundary: {delta_boundary}")
#         return fitness
#
#     def f2_v1(self, source_test_data, follow_up_test_data):
#         """
#         Metamorphic rule
#         """
#         def normalize(value):
#             return value / (value + 1)
#
#         # ref.: deep_test
#         original_label = source_test_data.orig_pred
#         original_pred = source_test_data.pred
#         transformed_pred = follow_up_test_data.pred
#         mse_orig = mean_squared_error(y_true=[original_label], y_pred=[original_pred])
#         mse_transformed = mean_squared_error(y_true=[original_label], y_pred=[transformed_pred])
#         consistent = True
#         if math.pow(original_label - transformed_pred, 2) <= self.lambda_ * mse_orig:
#             MR1 = ((self.lambda_ * mse_orig) - math.pow(original_label - transformed_pred, 2) + self.K)
#         else:
#             MR1 = - (math.pow(original_label - transformed_pred, 2) - (self.lambda_ * mse_orig))
#
#         if np.abs(mse_transformed - mse_orig) <= self.epsilon_:
#             MR2 = - ((self.epsilon_ - np.abs(mse_transformed - mse_orig)) + self.K)
#             consistent = False
#         else:
#             MR2 = (np.abs(mse_transformed - mse_orig) - self.epsilon_)
#
#         dist = max(MR1, MR2)
#         if dist < 0:
#             distance = 1 - normalize(abs(dist))
#         else:
#             distance = normalize(dist)
#
#         follow_up_test_data.consistent = consistent
#         print(f"--f2 : {distance}    consistent: {consistent}")
#
#         return distance
#
#     def f2_v2(self, source_test_data, follow_up_test_data):
#         """
#         Metamorphic rule
#         """
#         def normalize(value):
#             return value / (value + 1)
#
#         # ref.: deep_test
#         original_label = source_test_data.orig_pred
#         original_pred = source_test_data.pred
#         transformed_pred = follow_up_test_data.pred
#         mse_orig = mean_squared_error(y_true=[original_label], y_pred=[original_pred])
#         mse_transformed = mean_squared_error(y_true=[original_label], y_pred=[transformed_pred])
#         consistent = True
#         if math.pow(original_label - transformed_pred, 2) <= self.lambda_ * mse_orig:
#             MR1 = ((self.lambda_ * mse_orig) - math.pow(original_label - transformed_pred, 2) + self.K)
#         else:
#             MR1 = - (math.pow(original_label - transformed_pred, 2) - (self.lambda_ * mse_orig))
#
#         if np.abs(mse_transformed - mse_orig) <= self.epsilon_:
#             MR2 = 0
#             consistent = False
#         else:
#             MR2 = (np.abs(mse_transformed - mse_orig) - self.epsilon_)
#
#         v_unsafe = False
#         if MR1 < 0:
#             if MR1 >= -self.delta:
#                 v_unsafe = True
#                 dist = 0
#             else:
#                 dist = abs(MR1)
#         else:
#             dist = MR1
#
#         distance = dist + normalize(MR2)
#
#         follow_up_test_data.consistent = consistent
#         follow_up_test_data.v_unsafe = v_unsafe
#         print(f"--f2 : {distance}    consistent: {consistent}       v_unsafe: {v_unsafe}")
#
#         return distance
#
#     def f2_v3(self, source_test_data, follow_up_test_data):
#         """
#         Metamorphic rule
#         """
#         def normalize(value):
#             return value / (value + 1)
#
#         # ref.: deep_test
#         original_label = source_test_data.orig_pred
#         original_pred = source_test_data.pred
#         transformed_pred = follow_up_test_data.pred
#         mse_orig = mean_squared_error(y_true=[original_label], y_pred=[original_pred])
#         consistent = True
#         if math.pow(original_label - transformed_pred, 2) <= self.lambda_ * mse_orig:
#             MR1 = ((self.lambda_ * mse_orig) - math.pow(original_label - transformed_pred, 2) + self.K)
#         else:
#             MR1 = - (math.pow(original_label - transformed_pred, 2) - (self.lambda_ * mse_orig))
#             consistent = False
#
#         v_unsafe = False
#         # if MR1 < 0:
#         #     if MR1 >= -self.delta:
#         #         v_unsafe = True
#         #         distance = 0
#         #     else:
#         #         distance = abs(MR1)
#         # else:
#         #     distance = MR1
#
#         # version 2
#         if MR1 < 0:
#             if MR1 >= -self.delta:
#                 v_unsafe = True
#         distance = MR1
#
#         follow_up_test_data.consistent = consistent
#         follow_up_test_data.v_unsafe = v_unsafe
#         print(f"--f2 : {distance}    consistent: {consistent}       v_unsafe: {v_unsafe}")
#
#         return distance
#
#     def f2(self, source_test_data, follow_up_test_data):
#         """
#         Metamorphic rule
#         """
#         def normalize(value):
#             return value / (value + 1)
#
#         # ref.: deep_test
#         # original_label = source_test_data.orig_pred
#         original_pred = source_test_data.pred
#         transformed_pred = follow_up_test_data.pred
#
#         consistent = True
#         lhs = abs(original_pred - transformed_pred)
#         rhs = self.gama
#         if lhs <= rhs:
#             MR1 = rhs - lhs + self.K
#         else:
#             MR1 = - (lhs - rhs)
#             consistent = False
#
#         v_unsafe = False
#         # version 1
#         # if MR1 < 0:
#         #     if MR1 >= -self.delta:
#         #         v_unsafe = True
#         #         distance = 0
#         #     else:
#         #         distance = abs(MR1)
#         # else:
#         #     distance = MR1
#
#         # version 2
#         if MR1 < 0:
#             if MR1 >= -self.delta:
#                 v_unsafe = True
#         distance = MR1
#
#         follow_up_test_data.consistent = consistent
#         follow_up_test_data.v_unsafe = v_unsafe
#         print(f"--f2 : {distance}    consistent: {consistent}       v_unsafe: {v_unsafe}")
#
#         return distance
#
#     def f3(self, source_test_data, follow_up_test_data):
#         original_label = source_test_data.orig_pred
#         original_pred = source_test_data.pred
#         transformed_pred = follow_up_test_data.pred
#         mse_transformed = mean_squared_error(y_true=[original_label], y_pred=[transformed_pred])
#         mse_orig = mean_squared_error(y_true=[original_label], y_pred=[original_pred])
#         result = np.abs(mse_transformed - mse_orig) <= self.epsilon_
#
#         print(f"--f3 : {result} ")
#         return result


def image_contrast(img, params):
    print("image_contrast")
    if params < 0:
        return None
    alpha = params
    new_img = cv2.multiply(np.array(img), np.array([alpha]))
    return new_img

def image_blur(img, params):
    print("blur")
    img = np.array(img)
    blur = []
    if params == 0:
        return None
    if params == 1:
        blur = cv2.blur(img, (3, 3))
    if params == 2:
        blur = cv2.blur(img, (4, 4))
    if params == 3:
        blur = cv2.blur(img, (5, 5))
    if params == 4:
        blur = cv2.GaussianBlur(img, (3, 3), 0)
    if params == 5:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
    if params == 6:
        blur = cv2.GaussianBlur(img, (7, 7), 0)
    if params == 7:
        blur = cv2.medianBlur(img, 3)
    if params == 8:
        blur = cv2.medianBlur(img, 5)
    if params == 9:
        blur = cv2.blur(img, (6, 6))
    if params == 10:
        blur = cv2.bilateralFilter(img, 9, 75, 75)

    return blur

def flip(img, params):
    print("--image_flip")
    if params <= 0.5:
        new_img = None
    else:
        new_img = am.flipv(img)
    return new_img, "flip"

def brighten(img, params):
    print("--image_brighten")
    # if params <= 0:
    #     new_img = None
    # else:
    new_img = am.brighten(img, brightness_coeff=params)
    return new_img, "brighten"

def darken(img, params):
    print("--image_darken")
    # if params <= 0:
    #     new_img = None
    # else:
    new_img = am.darken(img, darkness_coeff=params)
    return new_img, "darken"

def add_snow(img, params):
    print("--image_add_snow")
    # if params <= 0:
    #     new_img = None
    # else:
    new_img = am.add_snow(img, snow_coeff=params)
    return new_img, "add_snow"

def add_rain_drizzle(img, params):
    print("--image_add_rain_drizzle")
    # if params == 0:
    #     new_img = None
    # else:
    new_img = am.add_rain(img, rain_type='drizzle', slant=params)
    return new_img, "add_rain_drizzle"

def add_rain_heavy(img, params):
    print("--image_add_rain_heavy")
    # if params == 0:
    #     new_img = None
    # else:
    new_img = am.add_rain(img, rain_type='heavy', slant=params)
    return new_img, "add_rain_heavy"

def add_rain_torrential(img, params):
    print("--image_add_rain_torrential")
    # if params == 0:
    #     new_img = None
    # else:
    new_img = am.add_rain(img, rain_type='torrential', slant=params)
    return new_img, "add_rain_torrential"

def add_fog(img, params):
    print("--image_add_fog")

    # if params <= 0:
    #     new_img = None
    # else:
    new_img = am.add_fog(img, fog_coeff=params)
    return new_img, "add_fog"

def add_sun_flare(img, params):
    print("--image_add_sun_flare")
    # if params <= 0:
    #     return None, _
    new_img = am.add_sun_flare(img, no_of_flare_circles=3, flare_center=(params, params))
    # hp.visualize(new_img, column=2)
    return new_img, "add_sun_flare"

def add_speed(img, params):
    print("--image_add_speed")
    # if params <= 0:
    #     return None, ""
    new_img = am.add_speed(img, speed_coeff=params)

    return new_img, "add_speed"

def composite_transform(img, params):
    if params[0] == 0:
        return None, _

    transform_id = params[0]-1
    feature = params[1]
    selected_transform = composite_transformations[transform_id]
    selected_param = selected_transform[2](
        feature * (selected_transform[1][1] - selected_transform[1][0]) +
        selected_transform[1][0])

    return selected_transform[0](img, selected_param)

def plot_pareto_frontier(Xs, Ys, save_path, maxX=True, maxY=False):
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)

    '''Plotting process'''
    plt.scatter(Xs, Ys)
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    plt.plot(pf_X, pf_Y)
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.ylim([min(Ys) - 0.1, max(Ys) + 0.1])
    # plt.show()
    plt.savefig(save_path)
    plt.close()

def reject_outliers(data, m=7.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    result = []
    indexes = []
    for index, x in enumerate(data):
        if s[index] < m:
            indexes.append(index)
            result.append(data[index])

    # return data[s < m].tolist()
    return indexes, result

def transformation_test():
    src_name = "1479425506394035143"
    param = -0.8190807656097758
    path = f"{SRC_DIR}/dataset/test/center/{src_name}.jpg"
    img = cv2.resize(cv2.imread(path), (100, 100))
    img_new = image_flip(img, param)
    cv2.imshow("xxx", img_new)
    cv2.waitKey(0)

def mutant_generator(model, source_test_data, follow_up_images_path, algorithm_name="NSGA2"):
    relative_dirctory = f"{follow_up_images_path}/{source_test_data.name}"
    fronts_info_file = f"{relative_dirctory}/fronts.obj"
    fronts_csv_file = f"{relative_dirctory}/fronts.csv"
    follow_ups_csv_file = f"{relative_dirctory}/follow_ups.csv"
    fronts_image = f"{relative_dirctory}/fronts.jpg"
    pathlib.Path(relative_dirctory).mkdir(parents=True, exist_ok=True)
    save_path = f"{relative_dirctory}/populations"
    log_path = f"{relative_dirctory}/populations.csv"
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    print("--setup mutant generator...")
    source_test_data.set_pred(model)
    source_test_data.extract_pathway(model)

    m = 10
    mse_orig = 0

    metrics = Metrics()
    algorithm = Algorithm(relative_dirctory)
    objectives = Objectives()

    num_of_variables = 0
    for x in transformations:
        if x[4]: #composit transformation
            num_of_variables += 2
        else:
            num_of_variables += 1

    problem = Problem(num_of_variables=num_of_variables,
                      objectives=[objectives.f1, objectives.f2],
                      validation=objectives.f3,
                      variables_range=[(0, 1)],
                      transformations=transformations,
                      source_test_data=source_test_data,
                      log_path=log_path,
                      model=model,
                      same_range=True
                      )
    utils = NSGA2(problem=problem,
                  num_of_individuals=10,
                  tournament_prob=0.9,
                  crossover_param=85,
                  mutation_param=75 #100 / num_of_variables
                  )

    if algorithm_name == "NSGA2":
        print("algorihtm: NSGA2")
        evo = Evolution(problem=problem,
                        save_path=save_path,
                        test_budget=60,  # in seconds
                        utils=utils,
                        algorithm=algorithm
                        )
        evo.register_on_new_generation(metrics.convergence)
        # evo.register_on_new_generation(metrics.record_hisotry)
        evo.register_on_new_generation(metrics.objective_space_animation)
        fronts = evo.evolve()
    else:
        print("algorihtm: RS")
        rs = RS(test_budget=60,
                generate_individual=problem.generate_individual,
                calculate_objectives=problem.calculate_objectives,
                fast_nondominated_sort=utils.fast_nondominated_sort)
        fronts = rs.run()

    first_time = True
    pareto_fronts = fronts[0]
    print("--save fronts...")
    with open(fronts_info_file, 'wb') as f:
        pickle.dump(pareto_fronts, f)

    pareto_fronts_objectives = [i.objectives for i in pareto_fronts]
    print("--plot pareto fronts...")
    Xs, Ys = zip(*pareto_fronts_objectives)
    # indexes, Xs = reject_outliers(np.array(Xs))
    # Ys = np.array(Ys)[indexes]
    # print(pareto_fronts_objectives)
    # print(Xs)
    # print(Ys)
    plot_pareto_frontier(Xs=Xs, Ys=Ys, save_path=fronts_image, maxY=False)

    #todo: fill fronts_info_file
    print(f"--#pareto fronts: {len(pareto_fronts)}")
    headres = ["features", "valid_transformations", "feasible", "delta_boundary", "consistent", "v_unsafe",
               "objectives"]
    init_log(fronts_csv_file, headres)
    for pareto in pareto_fronts:
        row = [pareto.features, pareto.name, pareto.feasible, pareto.delta_boundary,
               pareto.consistent, pareto.v_unsafe, pareto.objectives]
        log_step(fronts_csv_file, row)


    '''frontier selection process'''
    follow_ups = []
    i = 0
    headres = ["ID", "features", "valid_transformations", "feasible", "delta_boundary", "consistent", "v_unsafe",
               "objectives"]
    init_log(follow_ups_csv_file, headres)

    # version 1
    # while m > 0 and len(fronts[i]) > 0:
    #     pf = fronts[i]
    #     inconsistent_pareto_fronts = []
    #     consistent_pareto_fronts = []
    #     for pareto in pf:
    #
    #         if pareto.consistent is False:
    #             inconsistent_pareto_fronts.append(pareto)
    #         else:
    #             consistent_pareto_fronts.append(pareto)
    #
    #     # print("--pareto fronts logged")
    #     n_inconsistent = len(inconsistent_pareto_fronts)
    #     n_consistent = len(consistent_pareto_fronts)
    #
    #     sorted_inconsistent_pareto_fronts = sorted(inconsistent_pareto_fronts, key=lambda x: (x.objectives[1], x.objectives[0]),
    #                                                reverse=False)
    #
    #     if n_inconsistent >= m:
    #         follow_ups.extend(sorted_inconsistent_pareto_fronts[:m+1])
    #         break
    #     else:
    #         if n_inconsistent > 0:
    #             follow_ups.extend(sorted_inconsistent_pareto_fronts)
    #             m = m - len(follow_ups)
    #
    #         sorted_consistent_pareto_fronts = sorted(consistent_pareto_fronts, key=lambda x: (x.objectives[1], x.objectives[0]),
    #                                                    reverse=False)
    #         if n_consistent >= m:
    #             follow_ups.extend(sorted_consistent_pareto_fronts[:m+1])
    #             break
    #         else:
    #             follow_ups.extend(sorted_consistent_pareto_fronts)
    #             m = m - len(follow_ups)
    #     i += 1

    # version 2
    inconsistent_pareto_fronts = []
    consistent_pareto_fronts = []
    for pf in fronts:
        for pareto in pf:
            if pareto.consistent is False:
                inconsistent_pareto_fronts.append(pareto)
            else:
                consistent_pareto_fronts.append(pareto)

    # print("--pareto fronts logged")
    n_inconsistent = len(inconsistent_pareto_fronts)
    n_consistent = len(consistent_pareto_fronts)

    sorted_inconsistent_pareto_fronts = sorted(inconsistent_pareto_fronts, key=lambda x: (x.objectives[1], x.objectives[0]),
                                               reverse=False)

    if n_inconsistent >= m:
        follow_ups.extend(sorted_inconsistent_pareto_fronts[:m+1])
    else:
        if n_inconsistent > 0:
            follow_ups.extend(sorted_inconsistent_pareto_fronts)
            m = m - len(follow_ups)

        sorted_consistent_pareto_fronts = sorted(consistent_pareto_fronts, key=lambda x: (x.objectives[1], x.objectives[0]),
                                                   reverse=False)
        if n_consistent >= m:
            follow_ups.extend(sorted_consistent_pareto_fronts[:m+1])
        else:
            follow_ups.extend(sorted_consistent_pareto_fronts)

    follow_ups_names = []
    n_inconsistent = 0
    n_feasible = 0
    n_delta_boundary = 0
    n_v_unsafe = 0
    for i, follow_up in enumerate(follow_ups):
        row = [i, follow_up.features, follow_up.name, follow_up.feasible, follow_up.delta_boundary,
               follow_up.consistent, follow_up.v_unsafe, follow_up.objectives]
        log_step(follow_ups_csv_file, row)

        relative_path = follow_up_images_path + f"/{source_test_data.name}/follow_up_{i}.jpg"
        # copy_image(relative_path, follow_up_images_path)
        # image = Plot_tools.reverse_preprocess_imagenet_image(follow_up.tensor_data.clone())
        image = cv2.resize(follow_up.image_data, (100, 100))
        # cv2.imshow("front follow_up", image)
        # cv2.waitKey(1)
        # plt.close()
        cv2.imwrite(relative_path, image)

        follow_ups_names.append(follow_up.name)
        if not follow_up.consistent:
            n_inconsistent += 1
        if follow_up.feasible:
            n_feasible += 1
        if follow_up.delta_boundary:
            n_delta_boundary += 1
        if follow_up.v_unsafe:
            n_v_unsafe += 1

    print(f"--#inconsistent: {n_inconsistent}       #v_unsafe: {n_v_unsafe}   #feasible: {n_feasible}    #delta_boundary: {n_delta_boundary}")
    return follow_ups_names


composite_transformations = [
    (add_sun_flare, (0, 720), int, 0),
    (add_snow, (0, 1), float, 0),
    (add_rain_drizzle, (-10, 10), int, 0),
    (brighten, (0, 1), float, 0),
    (add_fog, (0, 1), float, 0),
    (add_speed, (0, 1), float, 0),
    (add_rain_torrential, (-10, 10), int, 0),
    (darken, (0, 1), float, 0),
    (add_rain_heavy, (-10, 10), int, 0)
]

composite_transformations_v2 = [
    (add_snow, (0, 0.1), float, 0),
    (add_rain_drizzle, (-0.1, 0.1), int, 0),
    (add_rain_heavy, (-0.1, 0.1), int, 0),
    (add_rain_torrential, (-0.1, 0.1), int, 0),
    (add_fog, (0, 0.1), float, 0),
    (add_sun_flare, (0, 50), int, 0),
    (flip, (-1, 0), int, -1),
    (brighten, (0, 0.1), float, 0),
    (add_speed, (0, 0.1), float, 0),
    (darken, (0, 0.1), float, 0),
]

composite_transformations_v1 = [
    (add_snow, (0, 0.01), float, 0),
    (add_rain_drizzle, (-0.01, 0.01), int, 0),
    (add_rain_heavy, (-0.01, 0.01), int, 0),
    (add_rain_torrential, (-0.01, 0.01), int, 0),
    (add_fog, (0, 0.01), float, 0),
    (add_sun_flare, (0, 50), int, 0),
    (flip, (-1, 0), int, -1),
    (brighten, (0, 0.01), float, 0),
    (add_speed, (0, 0.01), float, 0),
    (darken, (0, 0.01), float, 0)
]

transformations = [
    (flip, (0, 1), float, 0, False),
    (composite_transform, (1, len(composite_transformations)), int, 0, True)
]
if __name__ == "__main__":
    model_name = "dave_orig"
    epoch = 1
    source_image_path = f"{SRC_DIR}/dataset/test/center/1479425488640811418.jpg"
    src_path = SRC_DIR
    results_path = f"{src_path}/tools/DEEPDOMAIN/results/{model_name}/epoch_{epoch}"
    manifolds_path = f"{results_path}/manifolds"
    follow_up_images_path = f"{manifolds_path}/cache"
    weights_path = f"{SRC_DIR}/models/{model_name}/{model_name}.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    if model_name == "dave_orig":
        model = dave_orig()
    model.load_state_dict(torch.load(weights_path))
    _ = model.to(device).eval()

    source_test_data = Test_Data(image_path=source_image_path)
    source_test_data.prepare_data()
    self.name = os.path.basename(self.image_path).split(".")[0]
    tensor_data = prepare_data(source_image_path)
    source_test_data.pred = model(tensor_data)
    source_test_data.orig_pred = source_orig_pred

    follow_ups = mutant_generator(model, source_test_data, follow_up_images_path)
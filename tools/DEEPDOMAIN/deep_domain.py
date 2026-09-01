import pathlib
from turtle import pd
import np
# import cv2
# import faiss
# import pandas as pd
import sns

from constants import SRC_DIR
# import tool
# import torch
# from coverage import NBC
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.applications.imagenet_utils import preprocess_input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tools.DEEPDOMAIN.AI_self_driving_car_main.model.DAVE2 import DAVE2
from tools.DEEPDOMAIN.AI_self_driving_car_main.model.DriverNet import DriverNet
from tools.DEEPDOMAIN.AI_self_driving_car_main.model.TransferLearning import TLearning
# from torch.utils.data import DataLoader
# from torchvision import datasets, models, transforms
# from torchvision import transforms

import models.configs as model_configs
from models.utils.common import *
from tools.DEEPDOMAIN import tool
from tools.DEEPDOMAIN.coverage import NBC
from tools.DEEPDOMAIN.pymoo_mutant_generator import run_mutant_generator, random_mutant_generator, Test_Case, Test_Suite
from tools.DEEPDOMAIN.utils import *

class Deep_Domain:
    def __init__(self, directory):
        self.executed_tests = None
        self.directory = directory
        self.covered_sources = {}
        self.uncovered_classes = None
        self.classes_info = None
        self.source_executed_info_path = None
        self.follow_ups_executed_info_path = None
        self.database_path = None
        self.knowledge_base = {}
        self.init_variables()
        self.inconsistents_NBC = {}

    def init_variables(self):
        self.source_executed_info_path = f"{self.directory}/source_executed_info.csv"
        headers = ["source_image_path", "label", "yhat", "label_class_no", "pred_class_no"]
        init_log(self.source_executed_info_path, headers)
        #
        self.follow_ups_executed_info_path = f"{self.directory}/follow_ups_executed_info.csv"
        headers = ["iteration", "source_image_path", "follow_up_image_path", "label",  "yhat", "label_class_no", "pred_class_no"]
        init_log(self.follow_ups_executed_info_path, headers)

        self.database_path = f"{self.directory}/database.csv"
        headers = [f"x{i}" for i in range(n_variables)]
        headers.append("consistent")
        init_log(self.database_path, headers)


    def update_inconsistent_pool(self, main_directory, inconsistent_pool_path):
        cache_inconsistent_path = main_directory + "/inconsistents"
        counter = 0
        if os.path.exists(cache_inconsistent_path):
            for directory in os.listdir(cache_inconsistent_path):
                for file in os.listdir(cache_inconsistent_path + f"/{directory}"):
                    save_path = f"{inconsistent_pool_path}/{directory}"
                    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
                    src_file = cache_inconsistent_path + f"/{directory}/{file}"
                    move_file(src_file, save_path)
                    counter += 1

        return counter


    def update_inconsistents_NBC(self, inconsistent_pool_path, model, image_size):
        nc = 3
        input_size = (1, nc, image_size[0], image_size[1])
        random_data = torch.randn(input_size).to(device)
        model.eval()
        layer_size_dict = tool.get_layer_output_sizes(model, random_data)
        del random_data
        print("--Building range...")
        for directory in os.listdir(inconsistent_pool_path):
            torch.cuda.empty_cache()
            train_loader = get_data_loader(inconsistent_pool_path, str(directory), parameters.image_size, model_name)
            nbc = NBC(model, layer_size_dict, hyper=None, log_path=None)
            nbc.build(train_loader)
            self.inconsistents_NBC[int(directory)] = nbc


    def make_image_embeddings(self):
        feature_lists = []
        features_paths = f"{self.directory}/executed_images_features/"
        counter = 0
        for features_file in os.listdir(features_paths):
            features_norm = load_npy(features_paths + f"/{features_file}")
            feature_lists.append(features_norm)
            counter += 1
        self.executed_tests = np.vstack(feature_lists)

    def init_knowledge_base(self, model, n_iteration, source_list, classes_info, gama):
        self.uncovered_classes = list(source_list.keys())
        n_failures = 0
        for i in range(n_iteration):
            source_test_case = self.select_source_test_case(source_list, parameters.image_size)
            test_suite = Test_Suite(source_test_case, follow_up_images_path)
            copy_file(source_test_case.image_path, prediction_cache_path + "1.jpg")

            random_mutant_generator(test_suite, classes_info, self.database_path, model, model_name, inconsistent_pool_path,
                                    parameters, prediction_cache_path, default_label_file, device, gama)
            test_suite.save_image_fearures(self.directory, iteration)
            test_suite.log_executed_tests(self.directory, iteration)
            n_failure = self.update_inconsistent_pool(test_suite.directory, inconsistent_pool_path)
            n_failures += n_failure

        self.update_inconsistents_NBC(inconsistent_pool_path, model, parameters.image_size)
        self.make_image_embeddings()
        self.reset_uncovered_classes(source_list)
        return n_failures

    def save(self):
        save_npy(f"{self.directory}/executed_tests.npy", self.executed_tests)
    
    def add_test_case(self, test_case):
        if test_case.pred_class_no not in self.covered_sources.keys():
            self.covered_sources[test_case.pred_class_no] = [test_case.name]
        else:
            self.covered_sources[test_case.pred_class_no].append(test_case.name)

    def reset_uncovered_classes(self, source_list):
        if list(self.covered_sources.keys()) == list(source_list.keys()):
            self.uncovered_classes = list(source_list.keys())

    def load_source_test_cases(self, path, model, model_name):
        source_list_path = f"{path}/source_list_{model_name}.pkl"
        if os.path.exists(source_list_path):
            source_list = load_pickle(source_list_path)
        else:
            label_dataset = pd.read_csv(f"{path}/final_example.csv", dtype={'frame_id': 'str'}).values
            source_list = {}
            for index, row in enumerate(label_dataset):
                frame_id = row[0]
                label = np.float64(row[1])

                src_image_path = f"{source_images_path}/center/{frame_id}.jpg"
                dest_image_path = prediction_cache_path + "1.jpg"
                copy_file(src_image_path, dest_image_path)
                # print(frame_id)
                pred = model.predict_by_image_path(prediction_cache_path, default_label_file,
                                                        parameters, device)
                pred = np.float64(pred.item())
                pred_class_no = map_float_to_class(pred, classes_info)

                label_class_no = map_float_to_class(row[1], classes_info)
                info = [frame_id, label, pred, pred_class_no]
                if label_class_no not in source_list.keys():
                    source_list[label_class_no] = [info]
                else:
                    source_list[label_class_no].append(info)
            save_pickle(source_list_path, source_list)
        return source_list

    def select_source_test_case(self, source_list, image_size):
        uncovered_classes = self.uncovered_classes
        covered_sources = copy.deepcopy(self.covered_sources)

        # select an uncovered class
        if len(uncovered_classes) == 0:
            class_no = random.choices(list(source_list.keys()), k=1)[0]
        else:
            class_no = random.choices(uncovered_classes, k=1)[0]
            uncovered_classes.remove(class_no)
        # filter the class members
        if class_no not in covered_sources.keys():
            source_info = random.choices(source_list[class_no], k=1)[0]
            source_name = source_info[0]
            covered_sources[class_no] = [source_name]
        else:
            source_array = np.array(source_list[class_no], dtype=object)
            filtered_source_list = list(set(source_array[:, 0]) - set(covered_sources[class_no]))
            source_name = random.choices(filtered_source_list, k=1)[0]
            source_info = source_array[np.where(source_array[:, 0] == source_name)[0]][0]
            covered_sources[class_no].append(source_name)

        test_case = Test_Case()
        test_case.name = source_name
        test_case.label = source_info[1]
        test_case.label_class_no = class_no
        test_case.pred = source_info[2]
        test_case.pred_class_no = source_info[3]
        test_case.image_path = f"{source_images_path}/center/{source_name}.jpg"
        test_case.image_data = cv2.imread(test_case.image_path)

        test_case.extract_point()

        return test_case

if __name__ == "__main__":
    gama_list = [0.2, 0.3, 0.4]
    search_algorithm = "DNSGAII"
    sns.set(font_scale=1.5)
    model_list = ["dave2", "transfer", "DriverNet"]

    src_path = SRC_DIR
    source_images_path = f"{src_path}/dataset/test"

    prediction_cache_path = f"{src_path}/cache/center/"
    pathlib.Path(prediction_cache_path).mkdir(parents=True, exist_ok=True)

    default_label_file = f"{prediction_cache_path}/labels.csv"
    with open(default_label_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame_id", "steering_angle"])
        writer.writerow([1, -0.373665106110275])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model_name in model_list:
        for gama in gama_list:
            algorithm_seed = None
            mutant_generator_budget = 10 * 60  # seconds
            max_iteration = 6
            start_epoch = 1
            max_epoch = 6
            n_variables = 2
            m = 10

            if model_name == "dave2":
                parameters = model_configs.dave2_parameters
                model_object = DAVE2
            elif model_name == "transfer":
                parameters = model_configs.transfer_parameters
                model_object = TLearning
            else:
                parameters = model_configs.DriverNet_parameters
                model_object = DriverNet

            for epoch in range(start_epoch, max_epoch):

                test_budget = mutant_generator_budget * max_iteration
                iteration = 1
                termination_condition = False

                results_path = f"{src_path}/tools/DEEPDOMAIN/results/{model_name}/{search_algorithm}/final_gama_{gama}/epoch_{epoch}"
                console_log_path = f"{results_path}/log.txt"
                executed_images_features_path = f"{results_path}/executed_images_features"
                follow_up_images_path = f"{results_path}/cache"
                executed_images_path = f"{results_path}/executed_images/center"
                weights_path = f"{src_path}/models/{model_name}/{model_name}"
                inconsistent_pool_path = f"{results_path}/inconsistent_pool"
                pathlib.Path(follow_up_images_path).mkdir(parents=True, exist_ok=True)
                pathlib.Path(executed_images_features_path).mkdir(parents=True, exist_ok=True)
                pathlib.Path(executed_images_path).mkdir(parents=True, exist_ok=True)
                pathlib.Path(inconsistent_pool_path).mkdir(parents=True, exist_ok=True)

                print(f"*************Epoch: {epoch}***************")
                model = model_object()

                if parameters.model_name == 'transfer':
                    for param in model.ResNet.parameters():
                        param.requires_grad = False

                if model_name == "DriverNet":
                    model = torch.load(parameters.checkpoint, map_location=device)
                elif model_name == "TruckNN":
                    state = torch.load(parameters.checkpoint, map_location=device)['model_state_dict']
                    for key in list(state.keys()):
                        state[key.replace('module.', '')] = state.pop(key)
                    model.load_state_dict(state, strict=True)
                else:
                    model.load_state_dict(torch.load(parameters.checkpoint, map_location=device))

                _ = model.to(device).eval()

                classes_info = get_steering_classes()
                deep_domain = Deep_Domain(results_path)

                # step 1
                print("step 1: read all source test cases")
                source_list = deep_domain.load_source_test_cases(source_images_path, model, model_name)
                print("step 2: initialize the knowledge base...")
                start_time = time.time()
                n_failure = deep_domain.init_knowledge_base(model, n_iteration=10, source_list=source_list, classes_info=classes_info, gama=gama)
                print("#failures: ", n_failure)

                while not termination_condition:
                    print(f"*************Iteration: {iteration}***************")

                    # step 3
                    print(f"step 3: select {m} candidate source test cases and find the best...")
                    best_test_case = None
                    best_distance = -1.0
                    for i in range(m):
                        print(f"candidate {i}")
                        test_case = deep_domain.select_source_test_case(source_list, parameters.image_size)
                        mean_distance = test_case.calculate_distances(deep_domain.executed_tests)

                        if best_distance < mean_distance:
                            best_test_case = test_case
                            best_distance = mean_distance

                    # step 4
                    print("step 4: generate some follow-up test cases...")
                    test_suite = Test_Suite(best_test_case, follow_up_images_path)
                    copy_file(best_test_case.image_path, prediction_cache_path + "1.jpg")
                    remaining_budget = time.strftime("%H:%M:%S", time.gmtime((test_budget - (time.time() - start_time))))
                    algorithm_seed = run_mutant_generator(search_algorithm, model, model_name, test_suite, classes_info,
                                               deep_domain.database_path,
                                               inconsistent_pool_path,
                                               parameters, prediction_cache_path,
                                               default_label_file, device,
                                               inconsistents_NBC=deep_domain.inconsistents_NBC,
                                               time_budget=remaining_budget,
                                               gama=gama)
                    n_failure += deep_domain.update_inconsistent_pool(test_suite.directory, inconsistent_pool_path)
                    print("#failures: ", n_failure)
                    deep_domain.update_inconsistents_NBC(inconsistent_pool_path, model, parameters.image_size)
                    # step 5
                    print("step 5: save image features")
                    test_suite.save_image_fearures(deep_domain.directory, iteration)
                    deep_domain.make_image_embeddings()

                    # step 6
                    print("step 6: log executed tests")
                    test_suite.log_executed_tests(deep_domain.directory, iteration)
                    deep_domain.reset_uncovered_classes(source_list)

                    # step 7
                    print("step 7: add generated test suite to executed tests")
                    deep_domain.add_test_case(test_suite.source)

                    # step 8
                    print("step 8: update knowledge base...")
                    deep_domain.save()

                    # step 9
                    print("step 9: check termination condition")
                    termination_condition = test_budget < (time.time() - start_time)
                    iteration += 1


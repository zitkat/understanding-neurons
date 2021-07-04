import copy
import collections
import logging
import os
import numpy as np
import json
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})


class SafetyAnalysis:

    def __init__(self, MappedModel, EvalSet):
        super().__init__()

        self.feature_map_dict = {}
        self.fm_sum_responses_dict = {}
        self.statistics_dict = collections.defaultdict(list)

        self.EvalSet = EvalSet()
        self.MappedModel = MappedModel()
        self.original_layers = copy.deepcopy(self.MappedModel.renderable_layers)

        self.criticality_tau = 0.5

        log_file_dir = os.path.join('data', 'logging')
        if not os.path.exists(log_file_dir):
            os.makedirs(log_file_dir)

        self.logger = logging.getLogger('safety_utils')
        self.logger.setLevel(logging.DEBUG)

        today = "_{}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S_%f'))
        self.fh = logging.FileHandler(os.path.join(log_file_dir, str(today) + '_safety_utils_logger.log'))
        # create formatter and add it to the handlers
        self.formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(self.formatter)
        # add the handlers to the logger
        self.logger.addHandler(self.fh)

    def clear_variables(self):
        self.feature_map_dict = {}
        self.fm_sum_responses_dict = {}
        self.statistics_dict = collections.defaultdict(list)

    @staticmethod
    def get_layers_filter_position(_weights)-> int:
        if len(_weights.shape) == 4:  # to filter only depth separable and conv layers
            if _weights.shape[3] == 1:
                return 2
            else:
                return 3

        # to filter only dense layers
        elif len(_weights.shape) == 2:
           return 1

        else:
            return None

    @staticmethod
    def demask_filter_layer(_weights, _original_weights):
        _weights = _original_weights

    @staticmethod
    def mask_filter_layer(_weights, _indices):
        if len(_weights.shape) == 4:  # to filter only depth separable and conv layers

            if _weights.shape[3] == 1:
                for filters in _indices:
                    _weights[:, :, filters, :] = 0.0

            else:
                if _weights.shape[3] < max(_indices):
                    logging.error("Wrong filter index - layer contains less filters.")
                else:
                    for filters in _indices:
                        _weights[:, :, :, filters] = 0.0

        # to filter only dense layers
        elif len(_weights.shape) == 2:
           for neurons in _indices:
                _weights[:, neurons] = 0.0

    @staticmethod
    def calculate_criticality_adversary(des_adv_conf, new_conf, adv_ind, new_ind, ori_ind, adv_ori_conf, criticality_tau):
        criticality = 0.0

        # 1 case:
        """ if the new prediction has smaller confidence than original one """
        if new_ind != ori_ind and (des_adv_conf - adv_ori_conf) < criticality_tau:
            criticality = des_adv_conf - adv_ori_conf

        # 2 case:
        elif new_ind == ori_ind:
            if new_conf > 0.5:
                criticality = -2.0 # clip the criticality to 2
            else:
                criticality = -1.0 * (1 / (1 - (new_conf)))

        # 3 case:
        else:
            # Anti critical neurons, the criticality should be negative
            criticality = des_adv_conf - adv_ori_conf

        return criticality

    @staticmethod
    def calculate_criticality_classification(
            groundtruth_conf: float,
            masked_conf: float,
            goundtruth_index: int,
            masked_index: int,
            criticality_tau: float) -> float:

        criticality = 0.0

        # 1 case:
        """ if the new prediction has smaller confidence than original one """
        if ((groundtruth_conf - masked_conf) > criticality_tau) \
                and \
                (goundtruth_index == masked_index):
            criticality = groundtruth_conf - masked_conf

        # 2 case:
        elif goundtruth_index != masked_index:
            if masked_conf > 0.5:
                criticality = 2.0 # clip the criticality to 2
            else:
                criticality = 1 / (1 - masked_conf)

        # 3 case:
        else:
            # Anti critical neurons, the criticality should be negative
            criticality = groundtruth_conf - masked_conf

        return criticality

    @staticmethod
    def calculate_criticality_detection(
            groundtruth_iou: float,
            masked_iou: float) -> float:

        criticality = 0.0

        # 1 case: the masked IOU got smaller
        if groundtruth_iou > masked_iou:
            criticality = 2*(groundtruth_iou - masked_iou) # + 1
        # 2 case: the masked IOU got bigger
        else:
            criticality = 2*(groundtruth_iou - masked_iou) # + 1

        return criticality

    @staticmethod
    def calculate_resulting_criticality(
            criticality_cc: float,
            criticality_dc: float) -> float:

        return criticality_cc + criticality_dc

    def get_conf_and_class(self, des_cls: int, input_batch: dict, remove=False):

        # Get outputs from chosen layers and calculate maximum responses
        output = self.MappedModel(input_batch)
        if len(output.shape) > 1:
            output_layer = output[0]
        else:
            output_layer = output

        pre_ind = np.argmax(output_layer)
        pre_conf = np.max(output_layer)
        pre_cls = self.EvalSet.list_of_classes_labels[pre_ind]

        des_ind = self.EvalSet.list_of_classes_labels.index(des_cls)
        des_conf = output_layer[des_ind]

        #if ((des_cls != pre_cls) or (des_cls == pre_cls and pre_conf < 0.5)) and remove == True:
        #    os.remove(os.path.join( path, file_name))

        #self.logger.debug("Prediction index: {}, class: {}, confidence: {}, for image: {}".format(
        #                    pre_ind, pre_cls, pre_conf, file_name))

        return pre_ind, pre_conf, des_ind, des_conf

    def analyse_criticality_via_plain_masking(self):

        self.logger.debug(" ----------- Starting the CDPA_plain_masking ----------- ")

        statistics_dict_temp = collections.defaultdict(list)
        statistics_dict_json = collections.defaultdict(list)

        original_layers = copy.deepcopy(self.MappedModel.renderable_layers)
        masked_layers = self.MappedModel.renderable_layers

        for image_name, image, label in self.EvalSet.iterator():
            pre_ind, pre_conf, des_ind, des_conf = self.get_conf_and_class("hovno", image)
            logging.debug("Processing images: " + image_name)

            for original_layer, original_layers_name, masked_layers, masked_layers_name in zip(original_layers, masked_layers):
                logging.debug("Processing layer: " + original_layers_name)
                weights = masked_layers.weight.cpu().detach().numpy()
                indices = self.get_layers_filter_position(weights)

                for each_filter in tqdm(range(indices)):
                    # mask the related neuron
                    self.mask_filter_layer(weights, [each_filter])
                    output = self.MappedModel(image)
                    # outputs_dict[each_layer].append({str(each_filter): output})
                    if len(output.shape) > 1:
                        output_layer = output[0]
                    else:
                        output_layer = output
                    new_ind = np.argmax(output_layer)
                    new_conf = output_layer[new_ind]

                    criticality = self.calculate_criticality_classification(des_conf,
                                                                            new_conf,
                                                                            des_ind,
                                                                            new_ind,
                                                                            self.criticality_tau)

                    statistics_dict_temp[original_layers_name].append({str(each_filter): criticality})
                    statistics_dict_json[original_layers_name].append({str(each_filter): str(criticality)})

                # have to demask filter only at the end of the layer iteration
                self.demask_filter_layer(masked_layers, original_layer)

            self.logger.debug(" ----------- CDPA finished ----------- ")

            self.statistics_dict[image_name].append(statistics_dict_json)

        # dictionary_path = os.path.join("data", "statistics_dict" + _model_name + image_name + ".json")
        # with open(dictionary_path, 'w') as fp:
        #    json.dump(statistics_dict_json, fp)
        conv_dict = collections.defaultdict(list)
        proj_dict = collections.defaultdict(list)
        for layers_name in statistics_dict_temp.keys():
            if "project" in layers_name:
                for indices_dict in statistics_dict_temp[layers_name]:
                    proj_dict[layers_name].append(indices_dict)
            else:
                for indices_dict in statistics_dict_temp[layers_name]:
                    conv_dict[layers_name].append(indices_dict)

        #self.fun.plot_CDP_results(_path, proj_dict, image_name, _model_name, des_cls, self.dict_of_weights, self.criticality_tau, "project", _adversary=adversary)
        #self.fun.plot_CDP_results(_path, conv_dict, image_name, _model_name, des_cls, self.dict_of_weights, self.criticality_tau, "conv", _adversary=adversary)
        # self.fun.plot_layers_responses_results(_path, image_name, _model_name, des_cls, self.fm_sum_responses_dict)

    def analyse_CDP_plain_masking_offline(self, image_name):

        dictionary_path = os.path.join("data", "statistics_dict" + image_name + ".json")
        with open(dictionary_path, 'r') as fp:
            data = json.load(fp)
        print(data)

    def analyse_accuracy_of_masked_model(self, _files_path, _path_benchmark, _models, _classes, worst_neurons_dict, _n_worst=20):
        temp_worst_neurons = dict()

        for layer in worst_neurons_dict.keys():
            neurons_dicts = worst_neurons_dict[layer]
            for neurons_dict in neurons_dicts:
                for index, criticality in neurons_dict.items():
                    self.fun.setKey(temp_worst_neurons, criticality, [layer, index])

        sorted_worst_neurons = dict(sorted(temp_worst_neurons.items(), reverse=True))
        #print(sorted_worst_neurons)

        files = [x for x in os.listdir(_files_path) if not x.startswith('.')]
        files.sort()
        list_of_neurons = list(sorted_worst_neurons.keys())

        list_of_neurons = list_of_neurons[:_n_worst]
        list_of_acc = list()
        list_of_acc_std = list()
        list_of_worst_neurons = list()
        # first original accuracy
        acc, std_acc = self.calculate_accuracy(_files_path, files, _models, _classes)
        list_of_acc.append(acc)
        list_of_acc_std.append(std_acc)
        list_of_worst_neurons.append("No masking")

        for worst_neuron in list_of_neurons:
            layer = sorted_worst_neurons[worst_neuron][0][0]
            neuron_index = sorted_worst_neurons[worst_neuron][0][1]
            list_of_worst_neurons.append(layer + " " + str(neuron_index))
            # print(layer + " neuron: " + str(neuron_index))
            self.mask_filter_layer(layer, [int(neuron_index)], self.dict_of_layers, self.dict_of_weights)
            acc, std_acc = self.calculate_accuracy(_files_path, files, _models, _classes)
            list_of_acc.append(acc)
            list_of_acc_std.append(std_acc)
            self.demask_filter_layer(layer, self.dict_of_layers, self.dict_of_weights)

        fig_dir = os.path.join(_path_benchmark, _models, _classes)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        self.fun.plot_histogram_with_err(fig_dir, _models, _classes, _n_worst,
                                          list_of_worst_neurons, list_of_acc, list_of_acc_std)


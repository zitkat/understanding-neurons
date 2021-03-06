from __future__ import print_function, division

from PIL import Image
import os
import shutil
import numpy as np
import collections
import torch
from tqdm import tqdm


class DataSet:
    def __init__(self):
        self.list_of_classes_labels = self.open_imagenet_labels()
        self.num_classes = len(self.list_of_classes_labels)
        self.dataset = collections.defaultdict(list)

    def open_imagenet_labels(self):
        return self.open_txt_as_list(
            os.path.join("data", "dataset", "labels", "imagenet1000_clsidx_to_labels.txt"))

    @staticmethod
    def open_txt_as_list(path):
        l = []
        with open(path) as f:
            for line in f:
                (key, output_class, rest) = line.split("'", 2)
                l.append(output_class)
        return l

    def sort_and_copy_images_according_to_label(self, model, dataset_path, out_path, confidence_threshold, batch_size, device):
        valid_images = [".jpg", ".png", ".jpeg"]

        images = [x for x in os.listdir(dataset_path) if not x.startswith('.')]
        images.sort()

        for i in tqdm(range(0, len(images), batch_size)):
            input_batch = images[i:i + batch_size]

            images_batch = list()
            pre_ind = list()
            pre_conf = list()
            pre_cls = list()
            images_name = list()
            for image_name in input_batch:

                # import raw images and resize them
                ext = os.path.splitext(image_name)[-1]
                if ext.lower() in valid_images:
                    image = Image.open(os.path.join(dataset_path, image_name))
                    if image.layers > 2:
                        resized_image = image.resize(size=(224, 224), resample=Image.BILINEAR)
                        image_array = np.asarray(resized_image, dtype=np.float32)
                        image_array /= 255.
                        image_array = np.moveaxis(image_array, -1, 0)
                        images_batch.append(image_array)
                        images_name.append(image_name)

            if len(images_batch) > 0:
                images_array = np.asarray(images_batch)
                images_tensor = torch.FloatTensor(images_array)
                output = model(images_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                probabilities = probabilities.cpu().detach().numpy()

                pre_ind = list(np.argmax(probabilities, axis=1))
                pre_conf = list(np.max(probabilities, axis=1))
                pre_cls = [self.list_of_classes_labels[index] for index in pre_ind]

                for confidences, label, name_of_the_image in zip(pre_conf, pre_cls, images_name):
                    if confidences > confidence_threshold:
                        fig_dir = os.path.join(out_path, label)
                        if not os.path.exists(fig_dir):
                            os.makedirs(fig_dir)
                        shutil.copyfile(os.path.join(dataset_path, name_of_the_image), os.path.join(fig_dir, name_of_the_image))

        labels = [x for x in os.listdir(out_path) if not x.startswith('.')]
        labels.sort()
        for label in labels:
            list_of_images = [x for x in os.listdir(os.path.join(out_path, label)) if not x.startswith('.')]
            print("Label: " + label + ", contains: " + len(list_of_images))


    def load_testset_from_path(self,
                               _path_dataset,
                               _resize=True,
                               _normalize=True,
                               _channels_last=False,
                               _batch_size=144,
                               _width=224, _height=224, _channels=3):

        valid_images = [".jpg", ".png", "jpeg"]
        valid_numpy = [".npy", ".npz"]
        batch_index = 0
        folders = [x for x in os.listdir(_path_dataset) if not x.startswith('.')]
        folders.sort()
        files = collections.defaultdict(list)
        for folder in folders:
            images = [x for x in os.listdir(os.path.join(_path_dataset, folder)) if not x.startswith('.')]
            files[folder].append(images)
        # print(files)

        for class_name, file_names in files.items():
            batch_image = list()
            batch_label = list()

            for index, file_name in enumerate(file_names[0]):
                # import raw images and resize them
                ext = os.path.splitext(file_name)[-1]
                if ext.lower() in valid_images:
                    image = Image.open(os.path.join(_path_dataset, class_name, file_name))

                    if _resize:
                        resized_image = image.resize(size=(_height, _width), resample=Image.BILINEAR)
                        image_array = np.asarray(resized_image, dtype=np.float32)
                    else:
                        image_array = np.asarray(image, dtype=np.float32)

                    if _normalize:
                        image_array /= 255.

                    if not _channels_last:
                        image_array = np.moveaxis(image_array, -1, 0)

                    if _batch_size == 1:
                        self.dataset[file_name].append((image_array, class_name))
                    elif ((index + 1) % _batch_size) == 0:
                        self.dataset[str(batch_index)].append((np.asarray(batch_image), np.asarray(batch_label)))
                        batch_image = list()
                        batch_label = list()
                        batch_index += 1
                    else:
                        batch_image.append(image_array)
                        batch_label.append(class_name)

                # import numpy arrays - adversary attacks
                if ext.lower() in valid_numpy:
                    image = np.load(os.path.join(_path_dataset, class_name, file_name))
                    self.dataset[file_name].append((image, class_name))

        print("Total number of loaded images is {}".format(len(self.dataset.keys()) * _batch_size))

    def dataset_iterator(self):
        for image_name, data in self.dataset.items():
            for image, label in data:
                yield image_name, image, label

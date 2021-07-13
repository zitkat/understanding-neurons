from __future__ import print_function, division

from PIL import Image
import os
import numpy as np
import collections
import matplotlib.pyplot as plt


def set_key(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = [value]
    elif type(dictionary[key]) == list:
        dictionary[key].append(value)
    else:
        dictionary[key] = [dictionary[key], value]

class DataSet:
    def __init__(self):
        self.list_of_classes_labels = self.open_imagenet_labels()
        self.num_classes = len(self.list_of_classes_labels)
        self.dataset = collections.defaultdict(list)
        self.load_testset_from_path(
            os.path.join("data", "dataset", "images"),
            _resize=True,
            _normalize=True,
            _channels_last=False,
            _width=224,
            _height=224,
            _channels=3
        )

    def open_imagenet_labels(self):
        return self.open_txt_as_list(
            os.path.join("data", "dataset", "labels", "imagenet1000_clsidx_to_labels.txt"))

    @staticmethod
    def open_txt_as_list(path):
        l = []
        with open(path) as f:
            for line in f:
               (key, output_class, rest) = line.split("'",2)
               l.append(output_class)
        return l

    def load_testset_from_path(self,
                               _path_dataset,
                               _resize=True,
                               _normalize=True,
                               _channels_last=False,
                               _batch_size=144,
                               _width=224, _height=224, _channels=3):

        valid_images = [".jpg", ".png"]
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

            # @TBD: predict desired class and remove images with low confidence
            #if ((des_cls != pre_cls) or (des_cls == pre_cls and pre_conf < 0.5)) and remove:
            #    os.remove(os.path.join(path, file_name))

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
                    elif ((index+1)%_batch_size) == 0:
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


from __future__ import print_function, division

from PIL import Image
import os
import numpy as np


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
        self.images = self.load_testset_from_path(
            os.path.join("data", "dataset", "images"),
            _resize= True,
            _width= 224,
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

    def load_testset_from_path(self, _path_dataset, _resize=True, _width=224, _height=224, _channels=3):
        X = list()
        valid_images = [".jpg", ".png"]
        valid_numpy = [".npy", ".npz"]

        folders = [x for x in os.listdir(_path_dataset) if not x.startswith('.')]
        folders.sort()
        files = dict()
        for folder in folders:
            images = [x for x in os.listdir(os.path.join(_path_dataset,folder)) if not x.startswith('.')]
            set_key(files, folder, images)
        # print(files)

        for class_name, file_names in files.items():
            for file_name in file_names[0]:
                # import raw images and resize them
                ext = os.path.splitext(file_name)[-1]
                if ext.lower() in valid_images:
                    image = Image.open(os.path.join(_path_dataset, class_name, file_name))

                    if _resize:
                        rsize = image.resize(size=(_height, _width), resample=Image.BILINEAR)
                        image_array = np.asarray(rsize, dtype=np.float32)
                        image_array /= 255.
                        # print(image_array.shape)
                        X.append(image_array)
                # import numpy arrays - adversary attacks
                if ext.lower() in valid_numpy:
                    image = np.load(os.path.join(_path_dataset, class_name, file_name))
                    X.append(image)

        print("Total number of loaded images is {}".format(len(X)))

        return np.asarray(X, dtype=np.float32)


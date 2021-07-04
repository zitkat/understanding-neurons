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


def load_testset_from_path(_path_dataset, _resize=True, _width=224, _height=224, _channels=3):
    X = []
    valid_images = [".jpg", ".png"]
    valid_numpy = [".npy", ".npz"]

    files = [x for x in os.listdir(_path_dataset) if not x.startswith('.')]
    files.sort()
    # print(files)

    for file_name in files:

        # import raw images and resize them
        ext = os.path.splitext(file_name)[-1]
        if ext.lower() in valid_images:
            image = Image.open(os.path.join(_path_dataset, file_name))

            if _resize:
                rsize = image.resize(size=(_height, _width), resample=Image.BILINEAR)
                image_array = np.asarray(rsize, dtype=np.float32)
                image_array /= 255.
                # print(image_array.shape)
                X.append(image_array)
        # import numpy arrays - adversary attacks
        if ext.lower() in valid_numpy:
            image = np.load(os.path.join(_path_dataset, file_name))
            X.append(image)

    print("Total number of loaded images is {}".format(len(X)))

    return np.asarray(X, dtype=np.float32)


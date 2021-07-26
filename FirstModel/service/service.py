import os
import numpy as np
from PIL import Image


def import_images_and_assign_labels(
        folder, label, X, Y
):
    for file in os.listdir(folder):
        image_path = os.path.join(folder, file)
        im = Image.open(image_path)
        im = im.resize((52, 52))
        im = im.convert("RGB")
        im_arr = np.array(im)
        im_arr = np.reshape(im_arr, (52 * 52 * 3))
        X.append(im_arr)
        Y.append(label)

def import_dataset():

    dataset_folder = "C:\\Users\\bowet\\Documents\\projet_cdll\\dataSet\\"
    train_folder = os.path.join(dataset_folder, "train")
    test_folder = os.path.join(dataset_folder, "test")

    X_train = []
    y_train = []
    import_images_and_assign_labels(
        os.path.join(train_folder, "planete"), 1.0, X_train, y_train
    )
    import_images_and_assign_labels(
        os.path.join(train_folder, "pasplanete"), -1.0, X_train, y_train
    )
    X_test = []
    y_test = []
    import_images_and_assign_labels(
        os.path.join(test_folder, "planete"), 1.0, X_test, y_test
    )
    import_images_and_assign_labels(
        os.path.join(test_folder, "pasplanete"), -1.0, X_test, y_test
    )
    return (np.array(X_train) / 255.0, np.array(y_train)), \
           (np.array(X_test) / 255.0, np.array(y_test))


def import_dataset2():
    dataset_folder = "C:\\Users\\bowet\\Documents\\projet_cdll\\dataSet\\"
    train_folder = os.path.join(dataset_folder, "train")
    test_folder = os.path.join(dataset_folder, "test")

    X_train = []
    y_train = []
    import_images_and_assign_labels(
        os.path.join(train_folder, "gazeuses"), 1.0, X_train, y_train
    )
    import_images_and_assign_labels(
        os.path.join(train_folder, "telluriques"), -1.0, X_train, y_train
    )
    X_test = []
    y_test = []
    import_images_and_assign_labels(
        os.path.join(test_folder, "gazeuses"), 1.0, X_test, y_test
    )
    import_images_and_assign_labels(
        os.path.join(test_folder, "telluriques"), -1.0, X_test, y_test
    )
    return (np.array(X_train) / 255.0, np.array(y_train)), \
           (np.array(X_test) / 255.0, np.array(y_test))
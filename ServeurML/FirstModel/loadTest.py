import ctypes
import os
from ctypes import *

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def import_images_and_assign_labels(
        folder, label, X, Y
):
    for file in os.listdir(folder):
        image_path = os.path.join(folder, file)
        im = Image.open(image_path)
        im = im.resize((64, 64))
        im = im.convert("RGB")
        im_arr = np.array(im)
        im_arr = np.reshape(im_arr, (64 * 64 * 3))
        X.append(im_arr)
        Y.append(label)

def import_dataset():
    # dataset_folder = "C:/Users/Toky Cedric/Desktop/datasetPlanet"
    dataset_folder = "C:/Users/Toky Cedric/Desktop/Dataset"
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

def run():
    (X_train, y_train), (X_test, y_test) = import_dataset()

    path_to_dll = "C:/Users/Toky Cedric/Desktop/Etudes/Projet Annuel/CPPDLL_ForPython/cmake-build-debug/CPPDLL_ForPython.dll"
    mylib = cdll.LoadLibrary(path_to_dll)

    dataset_inputs = np.array(X_train)
    dataset_expected_outputs = np.array(y_train)

    init_tab = [2, 64, 1]
    init_size = len(init_tab)
    init_type = c_int * init_size
    init = init_type(*init_tab)
    mylib.create_mlp_model.argtypes = [init_type, c_int]
    mylib.create_mlp_model.restype = c_void_p

    model = mylib.create_mlp_model(init, int(init_size))

    mylib.getLengthX.argtypes = [c_void_p]
    mylib.restype = c_int
    tmp_len = mylib.getLengthX(model)
    test_dataset = X_test
    colors = ["blue" if output >= 0 else "red" for output in dataset_expected_outputs]

    flattened_dataset_inputs = []
    for p in dataset_inputs :
        flattened_dataset_inputs.append(p[0])
        flattened_dataset_inputs.append(p[1])

    # definition de train_classification_stochastic_gradient....
    arrsize_flat = len(flattened_dataset_inputs)
    arrtype_flat = c_float * arrsize_flat
    arr_flat = arrtype_flat(*flattened_dataset_inputs)

    arrsize_exp = len(dataset_expected_outputs)
    arrtype_exp = c_float * arrsize_exp
    arr_exp = arrtype_exp(*dataset_expected_outputs)

    mylib.train_classification_stochastic_gradient_backpropagation_mlp_model.argtypes = [c_void_p, arrtype_flat, c_int,
                                                                                         arrtype_exp, c_float, c_int]
    mylib.train_classification_stochastic_gradient_backpropagation_mlp_model.restype = None

    mylib.train_classification_stochastic_gradient_backpropagation_mlp_model(model, arr_flat, arrsize_flat, arr_exp,
                                                                             0.001, 10000)
    predicted_outputs = []
    for p in test_dataset :
        arrsizeP = len(p)
        arrtypeP = c_float * arrsizeP
        arrP = arrtypeP(*p)
        mylib.predict_mlp_model_classification.argtypes = [c_void_p, arrtypeP]
        mylib.predict_mlp_model_classification.restype = POINTER(c_float)
        tmp = []

        tmp = mylib.predict_mlp_model_classification(model, arrP)
        np_arr = np.ctypeslib.as_array(tmp, (tmp_len,))
        predicted_outputs.append(np_arr[0])

    predicted_outputs_colors = ['blue' if label >= 0 else 'red' for label in predicted_outputs]
    plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    plt.scatter([p[0] for p in dataset_inputs], [p[1] for p in dataset_inputs], c=colors, s=200)
    plt.show()

    flattened_dataset_inputs = []
    for p in dataset_inputs :
        flattened_dataset_inputs.append(p[0])
        flattened_dataset_inputs.append(p[1])


    print("predicted_outputs_: \n", predicted_outputs)
    mylib.savePMC.argtypes = [c_void_p, c_char_p]
    mylib.savePMC.restype = None
    mylib.savePMC(model, b'firstPMCmodel3')


    #
    # print("Sur le dataset de Train")
    # print(model.predict(X_train))
    #
    # print("Sur le dataset de Test")
    # print(model.predict(X_test))
    #
    # logs = model.fit(X_train, y_train, epochs=500,
    #           validation_data=(X_test, y_test),
    #                  batch_size=3)
    #
    # model.save("./model2")
    #
    # print("Sur le dataset de Train")
    # print(model.predict(X_train))
    #
    # print("Sur le dataset de Test")
    # print(model.predict(X_test))
    #
    #
    # model= keras.models.load_model('./model3')
    # print("Sur le dataset de Train")
    # print(model.predict(X_train))
    #
    # print("Sur le dataset de Test")
    # print(model.predict(X_test))
    #
    #
    # # plt.plot(logs.history['loss'], c="orange")
    # # plt.plot(logs.history['val_loss'], c="green")
    # # plt.show()

if __name__ == "__main__":
    run()
    print("hey")
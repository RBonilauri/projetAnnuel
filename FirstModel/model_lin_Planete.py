import os
from ctypes import *

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from service.service import import_dataset


def run():
    (X_train, y_train), (X_test, y_test) = import_dataset()
    path_to_dll = "C:/Users/33660/Documents/ESGI/Projet_annuel/etape3/cmake-build-debug/etape3.dll"
    mylib = cdll.LoadLibrary(path_to_dll)

    dataset_inputs = np.array(X_train)
    dataset_expected_outputs = np.array(y_train)
    mylib.create_linear_model.argtypes = [c_int]
    mylib.create_linear_model.restype = POINTER(c_float)

    model = mylib.create_linear_model(2)

    x = np.ctypeslib.as_array(model, (3,))

    test_dataset = X_test

    flattened_dataset_inputs = []
    for p in dataset_inputs :
        flattened_dataset_inputs.append(p[0])
        flattened_dataset_inputs.append(p[1])

    arr_size_flattened = len(flattened_dataset_inputs)
    arr_type_flattened = c_float * arr_size_flattened
    arr_flattened = arr_type_flattened(*flattened_dataset_inputs)

    arr_size_outputs = len(dataset_expected_outputs)
    arr_type_outputs = c_float * arr_size_outputs
    arr_outputs = arr_type_outputs(*dataset_expected_outputs)

    mylib.train_classification_rosenblatt_rule_linear_model.argtypes = [POINTER(c_float), arr_type_flattened,
                                                                        arr_type_outputs, c_float, c_int, c_int, c_int]
    mylib.train_classification_rosenblatt_rule_linear_model.restype = None
    mylib.train_classification_rosenblatt_rule_linear_model(model, arr_flattened, arr_outputs, 0.001, 10000, len(x),
                                                            len(flattened_dataset_inputs))
    result = []
    for p in test_dataset :
        arr_size_result = len(p)
        arr_type_result = c_float * arr_size_result
        arr_result = arr_type_result(*p)
        mylib.predict_linear_model_classification.argtypes = [POINTER(c_float), arr_type_result, c_int]
        mylib.predict_linear_model_classification.restype = c_float

        tmp = mylib.predict_linear_model_classification(model, arr_result, len(x))
        result.append(tmp)

    print(result)

    flattened_dataset_inputs = []
    for p in dataset_inputs :
        flattened_dataset_inputs.append(p[0])
        flattened_dataset_inputs.append(p[1])

    mylib.save_linear_model.argtypes = [POINTER(c_float), c_int]
    mylib.save_linear_model.restype = None
    mylib.save_linear_model(model, 2)

    mylib.destroy_linear_model.argtypes = [POINTER(c_float)]
    mylib.destroy_linear_model.restype = None
    mylib.destroy_linear_model(model)

if __name__ == "__main__":
    run()
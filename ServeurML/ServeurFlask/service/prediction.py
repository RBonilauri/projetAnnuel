import os
import pickle
from ctypes import *

import joblib
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def prediction_service(model, test_dataset):
    path_to_dll = "C:/Users/Toky Cedric/Desktop/Etudes/Projet Annuel/CPPDLL_ForPython/cmake-build-debug/CPPDLL_ForPython.dll"
    mylib = cdll.LoadLibrary(path_to_dll)

    dataset_inputs = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    dataset_expected_outputs = np.array([1, 1, -1, -1])
    colors = ["blue" if output >= 0 else "red" for output in dataset_expected_outputs]

    predicted_outputs = []
    for p in test_dataset :
        arrsizeP = len(p)
        arrtypeP = c_float * arrsizeP
        arrP = arrtypeP(*p)
        mylib.predict_mlp_model_classification.argtypes = [c_void_p, arrtypeP]
        mylib.predict_mlp_model_classification.restype = POINTER(c_float)
        tmp = []

        tmp = mylib.predict_mlp_model_classification(model, arrP)
        np_arr = np.ctypeslib.as_array(tmp, (1,))
        predicted_outputs.append(np_arr[0])

    print("here")
    predicted_outputs_colors = ['green' if label >= 0 else 'yellow' for label in predicted_outputs]
    plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    plt.scatter([p[0] for p in dataset_inputs], [p[1] for p in dataset_inputs], c=colors, s=200)
    plt.show()

def get_flat_image(im):
    X = []
    im = im.resize((32, 32))
    im = im.convert("RGB")
    im_arr = np.array(im)
    im_arr = np.reshape(im_arr, (32 * 32 * 3))
    X.append(im_arr)
    return np.array(X)/255.0




def get_prediction(model, image_test):
    path_to_dll = "C:/Users/Toky Cedric/Desktop/Etudes/Projet Annuel/CPPDLL_ForPython/cmake-build-debug/CPPDLL_ForPython.dll"
    mylib = cdll.LoadLibrary(path_to_dll)

    mylib.loadPMC.restype = c_void_p
    mylib.loadPMC.argtypes = [c_char_p]

    print(model.encode("utf-8"))
    model = mylib.loadPMC(model.encode("utf-8"))

    print(image_test)

    predicted_outputs = []
    for p in image_test :
        arrsizeP = len(p)
        arrtypeP = c_float * arrsizeP
        arrP = arrtypeP(*p)
        mylib.predict_mlp_model_classification.argtypes = [c_void_p, arrtypeP]
        mylib.predict_mlp_model_classification.restype = POINTER(c_float)
        tmp = []

        tmp = mylib.predict_mlp_model_classification(model, arrP)
        np_arr = np.ctypeslib.as_array(tmp, (1,))
        predicted_outputs.append(np_arr[0])

    print("after predict")
    print(type(predicted_outputs[0]))


    return predicted_outputs[0];


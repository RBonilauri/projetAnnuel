from ctypes import *
import matplotlib.pyplot as plt
import random
import numpy as np
from dataclasses import dataclass
import math

path_to_dll = "C:/Users/Toky Cedric/Desktop/Etudes/Projet Annuel/libML/cmake-build-debug/libML.dll"


def create_linear_model(input_dim: int) -> [float] :
    return [random.uniform(-1.0, 1.0) for i in range(input_dim + 1)]

if __name__ == "__main__":
    mylib = cdll.LoadLibrary(path_to_dll)

    mylib.create_linear_model.argtypes = [c_int]
    mylib.create_linear_model.restype = POINTER(c_float)

    result = mylib.create_linear_model(3)
    np_array = np.ctypeslib.as_array(result, (3,))
    tab = [1,2,3,4]
    print(tab[1:2])
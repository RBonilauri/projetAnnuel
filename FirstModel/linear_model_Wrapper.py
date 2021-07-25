from ctypes import *
import numpy as np


class MyLinearModelWrapper:
    def __init__(self, is_classification: bool = True
                 , alpha: float = 0.01, iterations_count: int = 10000):
        path_to_dll = "C:/Users/33660/Documents/ESGI/Projet_annuel/etape3/cmake-build-debug/etape3.dll"
        mylib = cdll.LoadLibrary(path_to_dll)

        mylib.create_linear_model.argtypes = [c_int]
        mylib.create_linear_model.restype = POINTER(c_float)

        self.model = mylib.create_linear_model(2)
        self.is_classification = is_classification
        self.alpha = alpha
        self.iterations_count = iterations_count
        # print("wrapper initialise")

    def fit(self, X, Y):
        path_to_dll = "C:/Users/33660/Documents/ESGI/Projet_annuel/etape3/cmake-build-debug/etape3.dll"
        mylib = cdll.LoadLibrary(path_to_dll)

        x = np.ctypeslib.as_array(self.model, (3,))

        # TODO: verfier X
        print("let's go")

        if self.is_classification:
            dataset_inputs = np.array(X)
            dataset_expected_outputs = np.array(Y)

            flattened_dataset_inputs = []
            for p in dataset_inputs:
                flattened_dataset_inputs.append(p[0])
                flattened_dataset_inputs.append(p[1])

            arr_size_flattened = len(flattened_dataset_inputs)
            arr_type_flattened = c_float * arr_size_flattened
            arr_flattened = arr_type_flattened(*flattened_dataset_inputs)

            arr_size_outputs = len(dataset_expected_outputs)
            arr_type_outputs = c_float * arr_size_outputs
            arr_outputs = arr_type_outputs(*dataset_expected_outputs)

            mylib.train_classification_rosenblatt_rule_linear_model.argtypes = [POINTER(c_float), arr_type_flattened,
                                                                                arr_type_outputs, c_float,
                                                                                c_int, c_int, c_int]
            mylib.train_classification_rosenblatt_rule_linear_model.restype = None
            mylib.train_classification_rosenblatt_rule_linear_model(self.model, arr_flattened,
                                                                    arr_outputs, self.alpha,
                                                                    self.iterations_count, len(x),
                                                                    len(flattened_dataset_inputs))

    def predict(self, X):
        path_to_dll = "C:/Users/33660/Documents/ESGI/Projet_annuel/etape3/cmake-build-debug/etape3.dll"
        mylib = cdll.LoadLibrary(path_to_dll)

        x = np.ctypeslib.as_array(self.model, (3,))

        results = []
        for p in X:
            if self.is_classification:
                arr_size_result = len(p)
                arr_type_result = c_float * arr_size_result
                arr_result = arr_type_result(*p)

                mylib.predict_linear_model_classification.argtypes = [POINTER(c_float), arr_type_result, c_int]
                mylib.predict_linear_model_classification.restype = c_float
                results.append(mylib.predict_linear_model_classification(self.model, arr_result, len(x)))

        return results

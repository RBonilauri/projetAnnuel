from ctypes import *
import numpy as np


class MLPWrapper() :

  def __init__(self, npl: [int], is_classification: bool = True,
               alpha: float = 0.01, iterations_count: int = 1000):

        path_to_dll = "C:/Users/Toky Cedric/Desktop/Etudes/Projet Annuel/CPPDLL_ForPython/cmake-build-debug/CPPDLL_ForPython.dll"
        mylib = cdll.LoadLibrary(path_to_dll)

        init_size = len(npl)
        init_type = c_int * init_size
        init = init_type(*npl)
        mylib.create_mlp_model.argtypes = [init_type, c_int]
        mylib.create_mlp_model.restype = c_void_p

        self.model = mylib.create_mlp_model(init, int(init_size))
        self.is_classification = is_classification
        self.alpha = alpha
        self.iterations_count = iterations_count
        print("wrapper initialisé")


  def fit(self, X, Y):
    path_to_dll = "C:/Users/Toky Cedric/Desktop/Etudes/Projet Annuel/CPPDLL_ForPython/cmake-build-debug/CPPDLL_ForPython.dll"
    mylib = cdll.LoadLibrary(path_to_dll)

    # if not hasattr(X, 'shape'):
    #   X = np.array(X)
    #
    # if len(X.shape) == 1:
    #   X = np.expand_dims(X, axis=0)
    #
    # if not hasattr(Y, 'shape'):
    #   Y = np.array(Y)
    #
    # if len(Y.shape) == 1:
    #   Y = np.expand_dims(Y, axis=0)

    if self.is_classification:
        dataset_inputs = np.array(X)
        dataset_expected_outputs = np.array(Y)

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


        mylib.train_classification_stochastic_gradient_backpropagation_mlp_model.argtypes = [c_void_p, arrtype_flat,
                                                                                             c_int,
                                                                                             arrtype_exp, c_float,
                                                                                             c_int]
        mylib.train_classification_stochastic_gradient_backpropagation_mlp_model.restype = None

        mylib.train_classification_stochastic_gradient_backpropagation_mlp_model(self.model, arr_flat, arrsize_flat, arr_exp,
                                                                                 self.alpha, self.iterations_count)


        #
        # mylib.train_classification_stochastic_gradient_backpropagation_mlp_model(self.model,
        #                                                                  X.flatten(),
        #                                                                  Y.flatten(),
        #                                                                  self.alpha,
        #                                                                  self.iterations_count)
    else:
      mylib.train_regression_stochastic_gradient_backpropagation_mlp_model(self.model,
                                                                     X.flatten(),
                                                                     Y.flatten(),
                                                                     self.alpha,
                                                                     self.iterations_count)




  def predict(self, X):
    path_to_dll = "C:/Users/Toky Cedric/Desktop/Etudes/Projet Annuel/CPPDLL_ForPython/cmake-build-debug/CPPDLL_ForPython.dll"
    mylib = cdll.LoadLibrary(path_to_dll)

    mylib.getLengthX.argtypes = [c_void_p]
    mylib.restype = c_int
    tmp_len = mylib.getLengthX(self.model)

    if not hasattr(X, 'shape'):
      X = np.array(X)

    if len(X.shape) == 1:
      X = np.expand_dims(X, axis=0)

    print("début de la prédiction")
    predicted_outputs = []
    for p in X:
      if self.is_classification:
          arrsizeP = len(p)
          arrtypeP = c_float * arrsizeP
          arrP = arrtypeP(*p)
          mylib.predict_mlp_model_classification.argtypes = [c_void_p, arrtypeP]
          mylib.predict_mlp_model_classification.restype = POINTER(c_float)
          tmp = mylib.predict_mlp_model_classification(self.model, arrP)
          np_arr = np.ctypeslib.as_array(tmp, (tmp_len,))
          predicted_outputs.append(np_arr[0])

    print("fin de la prédiction")
    return predicted_outputs

        # mylib.predict_linear_model_classification.argtypes = [POINTER(c_float), arr_type_result, c_int]
        # mylib.predict_linear_model_classification.restype = c_float
        # results.append(mylib.predict_mlp_model_classification(self.model,  p.flatten()))

    # return np.array(results)
import ctypes
from ctypes import *
import numpy as np
from matplotlib import pyplot as plt

from PMC import create_mlp_model, predict_mlp_model_classification, \
    train_classification_stochastic_gradient_backpropagation_mlp_model

if __name__ == "__main__":
    path_to_dll = "C:/Users/Toky Cedric/Desktop/Etudes/Projet Annuel/CPPDLL_ForPython/cmake-build-debug/CPPDLL_ForPython.dll"
    mylib = cdll.LoadLibrary(path_to_dll)

    dataset_inputs = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    dataset_expected_outputs = np.array([1, 1, -1, -1])

    init_tab = [2, 2, 1]
    init_size = len(init_tab)
    init_type = c_int * init_size
    init = init_type(*init_tab)

    # definition des fonctions :

    mylib.create_mlp_model.argtypes = [init_type, c_int]
    mylib.create_mlp_model.restype = c_void_p


    model = mylib.create_mlp_model(init, int(init_size))
    mylib.getLengthX.argtypes = [c_void_p]
    mylib.restype = c_int
    tmp_len = mylib.getLengthX(model)
    test_dataset = [[x1 / 10, x2 / 10] for x1 in range(-10, 20) for x2 in range(-10, 20)]
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
                                                                             0.001, 100000)
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



    mylib.test_str.restype = ctypes.c_char_p
    ret_string = mylib.test_str(b'google')
    print(ret_string.decode())

    mylib.savePMC.argtypes = [c_void_p, c_char_p]
    mylib.savePMC.restype = None
    mylib.savePMC(model, b'PMCmodel')

    # #Sérialisation du modèle pour utilisation future
    # joblib.dump(model, "./modelC++.joblib")
    #
    # with open('model_pickle','wb') as f:
    #     pickle.dump(model,f)
    #
    # with open('C:/Users/Toky Cedric/Desktop/ServeurML/FirstModel/model_pickle','rb') as f:
    #     mp = pickle.load(f)
    #
    # jmodel = joblib.load("./modelC++.joblib")
    #
    # print("model: ", model)
    # print("mp: ", mp)
    # print("jmodel:", jmodel)
    #
    # mylib.getLengthX.argtypes = [c_void_p]
    # mylib.restype = c_int
    # tmp_len = mylib.getLengthX(jmodel)
    # print("len: ",tmp_len)
    #
    # predicted_outputs = []
    # for p in test_dataset :
    #     arrsizeP = len(p)
    #     arrtypeP = c_float * arrsizeP
    #     arrP = arrtypeP(*p)
    #     mylib.predict_mlp_model_classification.argtypes = [c_void_p, arrtypeP]
    #     mylib.predict_mlp_model_classification.restype = POINTER(c_float)
    #     tmp = []
    #
    #     tmp = mylib.predict_mlp_model_classification(jmodel, arrP)
    #     np_arr = np.ctypeslib.as_array(tmp, (tmp_len,))
    #     predicted_outputs.append(np_arr[0])
    #
    # predicted_outputs_colors = ['green' if label >= 0 else 'yellow' for label in predicted_outputs]
    # plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    # plt.scatter([p[0] for p in dataset_inputs], [p[1] for p in dataset_inputs], c=colors, s=200)
    # plt.show()




    # dataset_inputs = [
    #     [0, 0],
    #     [1, 1],
    #     [0, 1],
    #     [1, 0],
    # ]
    #
    # dataset_expected_outputs = [
    #     -1,
    #     -1,
    #     1,
    #     1,
    # ]
    # model = create_mlp_model([2, 2, 1])
    # test_dataset = [[x1 / 10, x2 / 10] for x1 in range(-10, 20) for x2 in range(-10, 20)]
    #
    # colors = ["blue" if output >= 0 else "red" for output in dataset_expected_outputs]
    #
    # predicted_outputs = [predict_mlp_model_classification(model, p)[0] for p in test_dataset]
    # predicted_outputs_colors = ['blue' if label >= 0 else 'red' for label in predicted_outputs]
    # plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    # plt.scatter([p[0] for p in dataset_inputs], [p[1] for p in dataset_inputs], c=colors, s=200)
    # plt.show()
    #
    # flattened_dataset_inputs = []
    # for p in dataset_inputs :
    #     flattened_dataset_inputs.append(p[0])
    #     flattened_dataset_inputs.append(p[1])
    #
    # train_classification_stochastic_gradient_backpropagation_mlp_model(model,
    #                                                                    flattened_dataset_inputs,
    #                                                                    dataset_expected_outputs,
    #                                                                    alpha=0.001,
    #                                                                    iterations_count=100000)
    #
    # predicted_outputs = [predict_mlp_model_classification(model, p)[0] for p in test_dataset]
    # predicted_outputs_colors = ['blue' if label >= 0 else 'red' for label in predicted_outputs]
    # plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    # plt.scatter([p[0] for p in dataset_inputs], [p[1] for p in dataset_inputs], c=colors, s=200)
    # plt.show()
    #
    # flattened_dataset_inputs = []
    # for p in dataset_inputs :
    #     flattened_dataset_inputs.append(p[0])
    #     flattened_dataset_inputs.append(p[1])
    #
    # print("Sauvegarde du model")
    # #Sérialisation du modèle pour utilisation future
    # # joblib.dump(model, "./model.joblib")
    # with open('model_pickle','wb') as f:
    #     pickle.dump(model,f)
    #
    # with open('C:/Users/Toky Cedric/Desktop/ServeurML/FirstModel/model_pickle','rb') as f:
    #     mp = pickle.load(f)
    #
    # predicted_outputs = [predict_mlp_model_classification(mp,p)[0] for p in test_dataset]
    # predicted_outputs_colors = ['blue' if label >= 0 else 'red' for label in predicted_outputs]
    # plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    # plt.scatter([p[0] for p in dataset_inputs], [p[1] for p in dataset_inputs], c=colors, s=200)
    # plt.show()
    #
    # print("model: ", model)
    # print("mp: ", mp)
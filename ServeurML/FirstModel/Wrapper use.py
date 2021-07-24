from ctypes import cdll
import numpy as np

from MLP_Wrappe import MLPWrapper
from service.service import import_dataset1, import_dataset2


def run():
    (X_train, y_train), (X_test, y_test) = import_dataset2()
    path_to_dll = "C:/Users/Toky Cedric/Desktop/Etudes/Projet Annuel/CPPDLL_ForPython/cmake-build-debug/CPPDLL_ForPython.dll"
    mylib = cdll.LoadLibrary(path_to_dll)

    dataset_inputs = np.array(X_train)
    dataset_expected_outputs = np.array(y_train)

    init_tab = [2, 64, 1]

    print("initialisation du wrapper")
    wrapped_model = MLPWrapper(init_tab, iterations_count =len(X_train))

    print("début de l'entrainement")
    for epoch in range(100):
        wrapped_model.fit(X_train, y_train)
    print("Fin de l'entrainement")

    result = wrapped_model.predict(X_test)
    print(result)






if __name__ == "__main__":
    run()
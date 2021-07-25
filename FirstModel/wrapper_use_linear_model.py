from ctypes import cdll
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, mean_squared_error

from linear_model_Wrapper import MyLinearModelWrapper
from service.service import import_dataset1, import_dataset2


# def my_mean_squared_error(y_true, y_predicted):
#     return np.sum((y_true - y_predicted) ** 2) / len(y_true)


def run():
    (x_train, y_train), (x_test, y_test) = import_dataset1()
    path_to_dll = "C:/Users/33660/Documents/ESGI/Projet_annuel/etape3/cmake-build-debug/etape3.dll"
    mylib = cdll.LoadLibrary(path_to_dll)

    losses = []
    val_losses = []

    wraped_model = MyLinearModelWrapper()

    for epoch in range(10):
        wraped_model.fit(x_train, y_train)

        y_pred = wraped_model.predict((x_train))
        losses.append(mean_squared_error(y_train, y_pred))

        test_y_pred = wraped_model.predict((x_test))
        val_losses.append(mean_squared_error(y_test, test_y_pred))

    plt.plot(losses)
    plt.plot(val_losses)
    plt.show()

if __name__ == "__main__":
    run()

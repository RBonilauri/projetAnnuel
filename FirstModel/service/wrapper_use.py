from ctypes import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import confusion_matrix
from service import import_dataset
from wrapper import MLPWrapper


def run():

    (X_train, y_train), (X_test, y_test) = import_dataset()
    path_to_dll = "C:/Users/bowet/Documents/projet_cdll/cmake-build-debug/projet_cdll.dll"
    mylib = cdll.LoadLibrary(path_to_dll)

    dataset_inputs = np.array(X_train)
    dataset_expected_outputs = np.array(y_train)

    init_tab = [2, 8, 1]

    print("initialisation du wrapper")
    wrapped_model = MLPWrapper(init_tab, iterations_count =len(X_train))

    losses = []
    val_losses = []

    accs = []
    val_accs = []

    # print(len(y_train))
    # # print(y_test)
    print("début de l'entrainement")
    for epoch in range(1):
        wrapped_model.fit(X_train, y_train)

        y_pred = wrapped_model.predict(X_train)
        loss = mean_squared_error(y_train, y_pred)
        losses.append(loss)

        val_y_pred = wrapped_model.predict(X_test)
        val_loss = mean_squared_error(y_test, val_y_pred)
        val_losses.append(val_loss)

        acc_pred = []
        for p in y_pred:
            if p >= 0:
                acc_pred.append(1)
            else:
                acc_pred.append(-1)

        # print("y_train : ",y_train)
        # print("acc_pred : ",acc_pred)

        acc = accuracy_score(y_train, acc_pred)
        accs.append(acc)

        acc_val_y = []
        for p in val_y_pred:
            if p >= 0:
                acc_val_y.append(1)
            else:
                acc_val_y.append(-1)

        # print("y_test : ",y_test)
        # print("acc_val : ",acc_val_y)

        val_acc = accuracy_score(y_test, acc_val_y)
        val_accs.append(val_acc)

        print(f"epoch = {epoch} , loss = {loss}, val_loss = {val_loss}, acc = {acc}, val_acc = {val_acc}")

    print("Fin de l'entrainement")
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.title('Evolution of loss (MSE)')
    plt.xlabel('epochs')
    plt.ylabel(f'mean squared error')
    plt.show()

    plt.plot(accs)
    plt.plot(val_accs)
    plt.legend(['acc', 'val_acc'], loc='upper left')
    plt.title('Evolution of accuracy')
    plt.xlabel('epochs')
    plt.ylabel(f'accuracy')
    plt.show()

    result = wrapped_model.predict(X_test)
    acc_pred = []
    for p in result:
        if p >= 0:
            acc_pred.append(1)
        else:
            acc_pred.append(-1)


    confusion_matrix(y_test, acc_pred)
    print(result)

    mylib.savePMC.argtypes = [c_void_p, c_char_p]
    mylib.savePMC.restype = None
    mylib.savePMC(wrapped_model.model, b'teste_26/07')




if __name__ == "__main__":
    run()

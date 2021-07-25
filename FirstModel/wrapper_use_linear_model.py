from ctypes import cdll
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, mean_squared_error

from linear_model_Wrapper import MyLinearModelWrapper
from service.service import import_dataset1, import_dataset2


# def my_mean_squared_error(y_true, y_predicted):
#     return np.sum((y_true - y_predicted) ** 2) / len(y_true)


def run():
    (x_train, y_train), (x_test, y_test) = import_dataset2()
    path_to_dll = "C:/Users/33660/Documents/ESGI/Projet_annuel/etape3/cmake-build-debug/etape3.dll"
    mylib = cdll.LoadLibrary(path_to_dll)

    losses = []
    val_losses = []
    accs = []
    val_accs = []

    wrapped_model = MyLinearModelWrapper()

    for epoch in range(100):
        wrapped_model.fit(x_train, y_train)

        y_pred = wrapped_model.predict((x_train))
        losses.append(mean_squared_error(y_train, y_pred))

        test_y_pred = wrapped_model.predict((x_test))
        val_losses.append(mean_squared_error(y_test, test_y_pred))

        acc_pred = []
        for p in y_pred:
            if p >= 0:
                acc_pred.append(1)
            else:
                acc_pred.append(-1)

        acc = accuracy_score(y_train, acc_pred)
        accs.append(acc)

        acc_val_y = []
        for p in test_y_pred:
            if p >= 0:
                acc_val_y.append(1)
            else:
                acc_val_y.append(-1)

        val_acc = accuracy_score(y_test, acc_val_y)
        val_accs.append(val_acc)

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

    result = wrapped_model.predict(x_test)
    print("result :", result)


if __name__ == "__main__":
    run()

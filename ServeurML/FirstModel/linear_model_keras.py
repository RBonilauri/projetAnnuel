import os
import numpy as np
from PIL import Image
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from service.service import import_dataset1, import_dataset2


def run():
    (X_train, y_train), (X_test, y_test) = import_dataset1()

    model = keras.experimental.LinearModel()

    model.compile(optimizer=keras.optimizers.SGD(lr=1e-4), loss='mse', metrics=['accuracy'])

    print("Sur le dataset de Train")
    print(model.predict(X_train))

    print("Sur le dataset de Test")
    print(model.predict(X_test))

    logs = model.fit(X_train, y_train, epochs=400,
                     validation_data=(X_test, y_test),
                     batch_size=2)

    print("Sur le dataset de Train")
    print(model.predict(X_train))

    print("Sur le dataset de Test")
    print(model.predict(X_test))

    model.save("C:/Users/33660/CLionProjects/projetAnnuel789/ServeurML/Model/ML_save")

    plt.plot(logs.history['loss'], c="orange")
    plt.plot(logs.history['val_loss'], c="blue")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(logs.history['accuracy'], c="yellow")
    plt.plot(logs.history['val_accuracy'], c="red")
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    run()
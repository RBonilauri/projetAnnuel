import numpy as np
import tensorflow as tf
from tensorflow import keras

from service.service import import_dataset1, import_dataset2


def run():
    (X_train, y_train), (X_test, y_test) = import_dataset1()

    model = keras.models.load_model("C:/Users/33660/CLionProjects/projetAnnuel789/ServeurML/Model/ML_save")
    print("model :", model)
    print(model.predict(X_test))


if __name__ == "__main__":
    run()

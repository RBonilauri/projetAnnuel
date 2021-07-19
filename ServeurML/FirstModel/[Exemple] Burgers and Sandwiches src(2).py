import os
import numpy as np
from PIL import Image
import tensorflow.keras as keras
import matplotlib.pyplot as plt


def import_images_and_assign_labels(
        folder, label, X, Y
):
    for file in os.listdir(folder):
        image_path = os.path.join(folder, file)
        im = Image.open(image_path)
        im = im.resize((16, 16))
        im = im.convert("RGB")
        im_arr = np.array(im)
        im_arr = np.reshape(im_arr, (16 * 16 * 3))
        X.append(im_arr)
        Y.append(label)


def import_dataset():
    # dataset_folder = "C:/Users/Toky Cedric/Desktop/datasetPlanet"
    dataset_folder = "C:/Users/Toky Cedric/Desktop/Dataset"
    train_folder = os.path.join(dataset_folder, "train")
    test_folder = os.path.join(dataset_folder, "test")

    X_train = []
    y_train = []
    import_images_and_assign_labels(
        os.path.join(train_folder, "gazeuses"), 1.0, X_train, y_train
    )
    import_images_and_assign_labels(
        os.path.join(train_folder, "telluriques"), -1.0, X_train, y_train
    )
    X_test = []
    y_test = []
    import_images_and_assign_labels(
        os.path.join(test_folder, "gazeuses"), 1.0, X_test, y_test
    )
    import_images_and_assign_labels(
        os.path.join(test_folder, "telluriques"), -1.0, X_test, y_test
    )

    return (np.array(X_train) / 255.0, np.array(y_train)), \
           (np.array(X_test) / 255.0, np.array(y_test))


def run():
    (X_train, y_train), (X_test, y_test) = import_dataset()

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, activation=keras.activations.tanh))

    model.compile(optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
                  loss=keras.losses.mse)

    print("Sur le dataset de Train")
    print(model.predict(X_train))

    print("Sur le dataset de Test")
    print(model.predict(X_test))

    logs = model.fit(X_train, y_train, epochs=500,
              validation_data=(X_test, y_test),
                     batch_size=3)

    model.save("./model2")

    print("Sur le dataset de Train")
    print(model.predict(X_train))

    print("Sur le dataset de Test")
    print(model.predict(X_test))


    model= keras.models.load_model('./model3')
    print("Sur le dataset de Train")
    print(model.predict(X_train))

    print("Sur le dataset de Test")
    print(model.predict(X_test))


    # plt.plot(logs.history['loss'], c="orange")
    # plt.plot(logs.history['val_loss'], c="green")
    # plt.show()


if __name__ == "__main__":
    run()


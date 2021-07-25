import json
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, mean_squared_error
import pickle

from service.service import import_dataset1, import_dataset2


def run():
    (x_train, y_train), (x_test, y_test) = import_dataset1()

    # (x_train, y_train), (x_test, y_test) = import_dataset2()

    model = MLPClassifier()
    print("model :", model)

    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    print("train score :", train_score)

    model.fit(x_test, y_test)

    filename = 'MLP_sklearn.save'
    pickle.dump(model, open(filename, 'wb'))
    print("Model saved !")

    result = model.predict(x_test)
    print("result :", result)

    loaded_model = pickle.load(open(filename, 'rb'))
    test_score = loaded_model.score(x_test, y_test)
    print("test score :", test_score)


if __name__ == "__main__":
    run()

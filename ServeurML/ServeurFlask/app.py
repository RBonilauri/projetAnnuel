import ctypes
import json
import pickle
from ctypes import *
import numpy as np
from flask import Flask, request, jsonify
from matplotlib import pyplot as plt
import requests
from PIL import Image

from entities import PMC
import joblib
from service.prediction import prediction_service, get_flat_image, get_MLP_prediction, get_MLineaire_prediction

app = Flask(__name__)

@app.route('/test', methods=['GET'])
def testted():
    model_name = request.args.get("model")
    url = request.args.get("url")
    print("model_name: ", model_name)
    print("url: ", url)

    data = {
        "result": "hey"
    }

    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )

    return response


@app.route('/predictMLP', methods=['GET'])
def predictMLP() :
    model_name = request.args.get("model")
    url = request.args.get("url")

    model_name = model_name[1:len(model_name)-1]
    url = url[1:len(url)-1]

    im = Image.open(requests.get(url, stream=True).raw)
    image_for_test = get_flat_image(im)

    print(model_name)

    prediction = get_MLP_prediction(model_name, image_for_test)

    print("model_name: ", model_name)
    print("url: ", url)

    data = {
        "value" : json.dumps(str(prediction))
    }

    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )

    print("data :", data)
    print("response : ",response)

    return response

@app.route('/predictMLineaire', methods=['GET'])
def predictMLineaire() :
    model_name = request.args.get("model")
    url = request.args.get("url")

    model_name = model_name[1:len(model_name)-1]
    url = url[1:len(url)-1]

    im = Image.open(requests.get(url, stream=True).raw)
    image_for_test = get_flat_image(im)

    print(model_name)

    prediction = get_MLineaire_prediction(model_name, image_for_test)

    print("model_name: ", model_name)
    print("url: ", url)

    data = {
        "value" : json.dumps(str(prediction))
    }

    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )

    print("data :", data)
    print("response : ",response)

    return response

if __name__ == '__main__' :
    app.run()

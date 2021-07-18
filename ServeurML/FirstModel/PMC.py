import matplotlib.pyplot as plt
import random
import numpy as np
from dataclasses import dataclass
import math


@dataclass
class MLP() :
    W: [[[float]]]
    d: [int]
    X: [[float]]
    deltas: [[float]]

    def getD(self):
        return self.d

    def forward_pass(self, sample_inputs: [float], is_classification: bool) :
        L = len(self.d) - 1

        for j in range(1, self.d[0] + 1) :
            self.X[0][j] = sample_inputs[j - 1]

        for l in range(1, L + 1) :
            for j in range(1, self.d[l] + 1) :
                sum_result = 0.0
                for i in range(0, self.d[l - 1] + 1) :
                    sum_result += self.W[l][i][j] * self.X[l - 1][i]
                self.X[l][j] = sum_result
                if is_classification or l < L :
                    self.X[l][j] = math.tanh(self.X[l][j])

    def train_stochastic_gradient_backpropagation(self,
                                                  flattened_dataset_inputs: [float],
                                                  flattened_dataset_expected_outputs: [float],
                                                  is_classification: bool,
                                                  alpha: float = 0.001,
                                                  iterations_count: int = 100000) :
        input_dim = self.d[0]
        output_dim = self.d[-1]
        samples_count = len(flattened_dataset_inputs) // input_dim
        L = len(self.d) - 1

        for it in range(iterations_count) :
            k = random.randint(0, samples_count - 1)

            sample_input = flattened_dataset_inputs[k * input_dim :(k + 1) * input_dim]
            sample_expected_output = flattened_dataset_expected_outputs[k * output_dim :(k + 1) * output_dim]

            self.forward_pass(sample_input, is_classification)

            for j in range(1, self.d[L] + 1) :
                self.deltas[L][j] = (self.X[L][j] - sample_expected_output[j - 1])
                if is_classification :
                    self.deltas[L][j] *= (1 - self.X[L][j] * self.X[L][j])

            for l in reversed(range(1, L + 1)) :
                for i in range(1, self.d[l - 1] + 1) :
                    sum_result = 0.0
                    for j in range(1, self.d[l] + 1) :
                        sum_result += self.W[l][i][j] * self.deltas[l][j]

                    self.deltas[l - 1][i] = (1 - self.X[l - 1][i] * self.X[l - 1][i]) * sum_result

            for l in range(1, L + 1) :
                for i in range(0, self.d[l - 1] + 1) :
                    for j in range(1, self.d[l] + 1) :
                        self.W[l][i][j] -= alpha * self.X[l - 1][i] * self.deltas[l][j]


def create_mlp_model(npl: [int]) :
    W = []
    for l in range(len(npl)) :
        W.append([])
        if l == 0 :
            continue
        for i in range(npl[l - 1] + 1) :
            W[l].append([])
            for j in range(npl[l] + 1) :
                W[l][i].append(random.uniform(-1.0, 1.0))

    d = list(npl)

    X = []
    for l in range(len(npl)) :
        X.append([])
        for j in range(npl[l] + 1) :
            X[l].append(1.0 if j == 0 else 0.0)

    deltas = []
    for l in range(len(npl)) :
        deltas.append([])
        for j in range(npl[l] + 1) :
            deltas[l].append(0.0)

    return MLP(W, d, X, deltas)

def predict_mlp_model_regression(model: MLP, sample_inputs:[float])-> [float]:
  model.forward_pass(sample_inputs, False)
  return model.X[-1][1:]

def predict_mlp_model_classification(model: MLP, sample_inputs:[float])-> [float]:
  model.forward_pass(sample_inputs, True)
  return model.X[-1][1:]

def train_classification_stochastic_gradient_backpropagation_mlp_model(model: MLP,
                                                                       flattened_dataset_inputs: [float],
                                                                       flattened_dataset_expected_outputs: [float],
                                                                       alpha: float = 0.001,
                                                                       iterations_count: int = 100000):
  model.train_stochastic_gradient_backpropagation(flattened_dataset_inputs,
                                                  flattened_dataset_expected_outputs,
                                                  True,
                                                  alpha,
                                                  iterations_count)


def train_regression_stochastic_gradient_backpropagation_mlp_model(model: MLP,
                                                                       flattened_dataset_inputs: [float],
                                                                       flattened_dataset_expected_outputs: [float],
                                                                       alpha: float = 0.001,
                                                                       iterations_count: int = 100000):
  model.train_stochastic_gradient_backpropagation(flattened_dataset_inputs,
                                                  flattened_dataset_expected_outputs,
                                                  False,
                                                  alpha,
                                                  iterations_count)
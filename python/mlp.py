from . import c_wrapper
import numpy as np

class MLP:
    def __init__(self, layer_sizes, activation="ReLu", loss="MSE", learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.loss = loss
        self.learning_rate = learning_rate

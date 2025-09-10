from . import c_wrapper
import numpy as np

class MLP:
    def __init__(self, layer_sizes: List[int], activation="Sigmoid", str: loss="MSE", learning_rate=0.01, seed: Optional[int]):

        if len(layer_sizes) < 2:
            print("Layer sizes must be >= 2")

        self.layer_sizes = layer_sizes
        self.activation = activation
        self.loss = loss
        self.learning_rate = learning_rate

        self.weights = [] # List of C matrix pointers
        self.biases = [] # List of C matrix pointers


import c_wrapper
import numpy as np
from typing import List, Optional

class MLP:
    def __init__(self, layer_sizes: List[int], activation="Sigmoid", loss: str="MSE", learning_rate=0.01, seed: Optional[int]=None):
        if len(layer_sizes) < 2:
            print("Layer sizes must be >= 2")

        self.layer_sizes = layer_sizes
        self.activation = activation
        self.loss = loss
        self.learning_rate = learning_rate
        self.activations = [] # will be populated in loops

        self.weights = [] # List of C matrix pointers
        self.biases = [] # List of C matrix pointer

        for i in range(len(self.layer_sizes) - 1):
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            w_np = np.random.uniform(-limit, limit, (layer_sizes[i+1], layer_sizes[i]))
            
            b_np = np.zeros((layer_sizes[i+1], 1))

            self.weights.append(c_wrapper.from_numpy(w_np))
            self.biases.append(c_wrapper.from_numpy(b_np))

    #When MLP is Garbage Collected, free the C allocated memory for the weights and biases
    def __del__(self):
        for w in self.weights:
            c_wrapper.free_py_matrix(w)
        for b in self.biases:
            c_wrapper.free_py_matrix(b)

    def forward_pass(self, X: c_wrapper.Matrix):
        """
        Compute the forward pass between two layers
        """
        self.activations = [X]
        cur_output = X

        for layer_index in range(len(self.weights)):
            weights = self.weights[layer_index]
            biases = self.biases[layer_index]

            input_matrix = c_wrapper.add_py_matrices(c_wrapper.multiply_py_matrices(weights, cur_output), biases) 

            if self.activation == "Sigmoid":
                activated_matrix = c_wrapper.py_sigmoid(input_matrix)
            else:
                raise "The activation function {self.activation} has not been implemented"

            c_wrapper.free_py_matrix(input_matrix)

            cur_output = activated_matrix
            self.activations.append(cur_output)

        return cur_output 

    def backward_pass(self, X: c_wrapper.Matrix, y: c_wrapper.Matrix):
        """
        Compute the backwards pass for the whole network
        """
        if self.loss == "MSE":
            output_error = py_mean_squared_error(y_true, y_pred, size)
        else:
            print(f"{self.loss} has not been defined yet. Please initalise with a different loss function")
        
        for layer_index in range(len(self.weights)):
            weights = self.weights[layer_index]
            biases = self.biases[layer_index]
            hidden_error = scalar_multiply_py_matrix(output_error, weights) * py_sigmoid_derivative()

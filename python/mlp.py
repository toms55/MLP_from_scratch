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
        Compute the forward pass for the whole network 
        """
        self.activations = [X]
        cur_output = X

        for layer_index in range(len(self.weights)):
            weights = self.weights[layer_index]
            biases = self.biases[layer_index]

            weights_matrix = c_wrapper.multiply_py_matrices(weights, cur_output)
            input_matrix = c_wrapper.py_add_weights_and_biases(weights_matrix, biases)

            if self.activation == "Sigmoid":
                activated_matrix = c_wrapper.py_sigmoid(input_matrix)
            else:
                raise "The activation function {self.activation} has not been implemented"

            c_wrapper.free_py_matrix(input_matrix)

            cur_output = activated_matrix
            self.activations.append(cur_output)

        return cur_output 

    def backward_pass(self, X: c_wrapper.Matrix, y_true: c_wrapper.Matrix, y_pred: c_wrapper.Matrix):
        if self.loss == "MSE":
            # derivative = 1/N(y_pred - y_true)
            diff = c_wrapper.subtract_py_matrices(y_pred, y_true)
            initial_loss_grad = c_wrapper.scalar_multiply_py_matrix(diff, 1 / y_true.rows)
            c_wrapper.free_py_matrix(diff)
        else:
            raise ValueError(f"{self.loss} has not been defined yet.")
        
        output_error = initial_loss_grad
        
        for layer_index in range(len(self.weights) - 1, -1, -1):
            weights = self.weights[layer_index]
            biases = self.biases[layer_index]
            activations = self.activations[layer_index + 1]
            prev_activation = self.activations[layer_index]
            
            activation_derivative = c_wrapper.py_sigmoid_derivative(activations) # make this conditional based on self.activation

            if layer_index == len(self.weights) - 1:
                error_signal = c_wrapper.hadamard_py_matrices(output_error, activation_derivative)
            else:
                next_weights = self.weights[layer_index + 1]
                transposed_weights = c_wrapper.transpose_py_matrix(next_weights)
                backpropagate = c_wrapper.multiply_py_matrices(transposed_weights, output_error)
                error_signal = c_wrapper.hadamard_py_matrices(backpropagate, activation_derivative)
                c_wrapper.free_py_matrix(transposed_weights)
                c_wrapper.free_py_matrix(backpropagate)

            c_wrapper.free_py_matrix(activation_derivative)
            
            transposed_prev_activation = c_wrapper.transpose_py_matrix(prev_activation)
            weight_gradient = c_wrapper.multiply_py_matrices(error_signal, transposed_prev_activation)
            bias_gradient = error_signal # this needs to be fixed so that error signal has the same dimensions as bias gradient
            
            scaled_wg = c_wrapper.scalar_multiply_py_matrix(weight_gradient, self.learning_rate)
            scaled_bg = c_wrapper.scalar_multiply_py_matrix(bias_gradient, self.learning_rate)
            
            new_weights = c_wrapper.subtract_py_matrices(weights, scaled_wg)
            new_biases = c_wrapper.subtract_py_matrices(biases, scaled_bg)
            
            c_wrapper.free_py_matrix(self.weights[layer_index])
            c_wrapper.free_py_matrix(self.biases[layer_index])

            self.weights[layer_index] = new_weights
            self.biases[layer_index] = new_biases

            c_wrapper.free_py_matrix(transposed_prev_activation)
            c_wrapper.free_py_matrix(weight_gradient)
            c_wrapper.free_py_matrix(scaled_wg)
            c_wrapper.free_py_matrix(scaled_bg)
            
            c_wrapper.free_py_matrix(output_error)
            output_error = error_signal

        c_wrapper.free_py_matrix(output_error)

    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int):
        X = c_wrapper.transpose_py_matrix(c_wrapper.from_numpy(X))
        y = c_wrapper.transpose_py_matrix(c_wrapper.from_numpy(y))

        for epoch in range(epochs):
            print(f"Training epoch {epoch}\n")

            y_pred = self.forward_pass(X)
            
            loss = c_wrapper.py_mean_squared_error(y, y_pred)
            print(f"This epoch's loss is {loss:.6f}")

            self.backward_pass(X, y, y_pred)

        print("Training Complete")

    def predict(self, X: np.ndarray):
        X = c_wrapper.transpose_py_matrix(c_wrapper.from_numpy(X))
        y_pred = c_wrapper.transpose_py_matrix(c_wrapper.from_numpy(y_pred))

        prediction = c_wrapper.to_numpy(y_pred)

        return prediction

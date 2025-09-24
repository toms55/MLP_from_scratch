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
        self.activations = []

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

    def _clear_activations(self):
        # The first activation is the input X, which is managed outside this class.
        # We only free the matrices created during the forward pass.
        for i in range(1, len(self.activations)):
            c_wrapper.free_py_matrix(self.activations[i])
        self.activations = []

    def forward_pass(self, X: c_wrapper.Matrix):
        """
        Compute the forward pass for the whole network 
        """
        self._clear_activations()

        self.activations = [X]
        cur_output = X

        for layer_index in range(len(self.weights)):
            weights = self.weights[layer_index]
            biases = self.biases[layer_index]

            weights_matrix = c_wrapper.multiply_py_matrices(weights, cur_output)
            input_matrix = c_wrapper.py_add_weights_and_biases(weights_matrix, biases)

            c_wrapper.free_py_matrix(weights_matrix)

            if self.activation == "Sigmoid":
                activated_matrix = c_wrapper.py_sigmoid(input_matrix)
            else:
                raise ValueError(f"The activation function {self.activation} has not been implemented")

            c_wrapper.free_py_matrix(input_matrix)

            cur_output = activated_matrix
            self.activations.append(cur_output)

        return cur_output 

    def backward_pass(self, y_true: c_wrapper.Matrix, y_pred: c_wrapper.Matrix):
        # Compute initial loss gradient
        if self.loss != "MSE":
            raise ValueError(f"{self.loss} has not been defined yet.")
        
        diff = c_wrapper.subtract_py_matrices(y_pred, y_true)
        output_error = c_wrapper.scalar_multiply_py_matrix(diff, 2 / (y_true.cols * y_true.rows))
        c_wrapper.free_py_matrix(diff)
        
        # Backpropagate through layers
        for layer_index in range(len(self.weights) - 1, -1, -1):
            activation_derivative = c_wrapper.py_sigmoid_derivative(self.activations[layer_index + 1])
            
            # Compute error signal
            if layer_index == len(self.weights) - 1:
                error_signal = c_wrapper.hadamard_py_matrices(output_error, activation_derivative)
            else:
                transposed_weights = c_wrapper.transpose_py_matrix(self.weights[layer_index + 1])
                backpropagate = c_wrapper.multiply_py_matrices(transposed_weights, output_error)
                error_signal = c_wrapper.hadamard_py_matrices(backpropagate, activation_derivative)
                c_wrapper.free_py_matrix(transposed_weights)
                c_wrapper.free_py_matrix(backpropagate)
            
            # Compute gradients and update parameters
            transposed_prev = c_wrapper.transpose_py_matrix(self.activations[layer_index])
            weight_grad = c_wrapper.scalar_multiply_py_matrix(c_wrapper.multiply_py_matrices(error_signal, transposed_prev), self.learning_rate)
            bias_grad = c_wrapper.scalar_multiply_py_matrix(c_wrapper.sum_py_matrix_columns(error_signal), self.learning_rate)
            
            # Update parameters
            new_weights = c_wrapper.subtract_py_matrices(self.weights[layer_index], weight_grad)
            new_biases = c_wrapper.subtract_py_matrices(self.biases[layer_index], bias_grad)

            c_wrapper.free_py_matrix(self.weights[layer_index])
            c_wrapper.free_py_matrix(self.biases[layer_index])

            self.weights[layer_index] = new_weights
            self.biases[layer_index] = new_biases

            for temp in [activation_derivative, transposed_prev, weight_grad, bias_grad]:
                c_wrapper.free_py_matrix(temp)
            
            # Propagate error
            if output_error.c_ptr != error_signal.c_ptr:
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

            self.backward_pass(y, y_pred)

            self._clear_activations()

        c_wrapper.free_py_matrix(X)
        c_wrapper.free_py_matrix(y)

        print("Training Complete")

    def predict(self, X: np.ndarray):
        X = c_wrapper.transpose_py_matrix(c_wrapper.from_numpy(X))

        y_pred = self.forward_pass(X)
        y_pred_transposed = c_wrapper.transpose_py_matrix(y_pred)

        prediction = c_wrapper.to_numpy(y_pred_transposed)

        c_wrapper.free_py_matrix(X)
        c_wrapper.free_py_matrix(y_pred_transposed)
        # Note: The memory for y_pred_c will be freed by the _clear_activations()

        return prediction

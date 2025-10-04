import c_wrapper
import numpy as np
from typing import List, Optional

class MLP:
    def __init__(self, layer_sizes: List[int], hidden_activation: str, output_activation: str, loss: str="MSE", learning_rate=0.01, seed: Optional[int]=None):
        if len(layer_sizes) < 2:
            print("Layer sizes must be >= 2")

        self.layer_sizes = layer_sizes
        self.hidden_activation = hidden_activation.lower()
        self.output_activation = output_activation.lower()
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

    def _clear_activations(self):
        for i in range(1, len(self.activations)):
            c_wrapper.free_py_matrix(self.activations[i])
        self.activations = []

    def forward_pass(self, X: c_wrapper.Matrix):
        self._clear_activations()

        self.activations = [X]
        cur_output = X
        num_layers = len(self.weights)

        for layer_index in range(num_layers):
            weights = self.weights[layer_index]
            biases = self.biases[layer_index]

            weights_matrix = c_wrapper.multiply_py_matrices(weights, cur_output)
            input_matrix = c_wrapper.py_add_weights_and_biases(weights_matrix, biases)

            c_wrapper.free_py_matrix(weights_matrix)

            is_output_layer = (layer_index == num_layers - 1)
            activation_type = self.output_activation if is_output_layer else self.hidden_activation
            
            if activation_type == "sigmoid":
                activated_matrix = c_wrapper.py_sigmoid(input_matrix)
                c_wrapper.free_py_matrix(input_matrix)
            elif activation_type == "relu":
                activated_matrix = c_wrapper.py_matrix_relu(input_matrix)
                c_wrapper.free_py_matrix(input_matrix)
            elif activation_type == "identity":
                activated_matrix = input_matrix
            else:
                raise ValueError(f"The activation function {activation_type} has not been implemented")

            cur_output = activated_matrix
            self.activations.append(cur_output)

        return cur_output 

    def backward_pass(self, y_true: c_wrapper.Matrix, y_pred: c_wrapper.Matrix):
        if self.loss != "MSE":
            raise ValueError(f"{self.loss} has not been defined yet.")
        
        # Gradient for MSE: 2 * (y_pred - y_true) / N
        diff = c_wrapper.subtract_py_matrices(y_pred, y_true)
        output_error = c_wrapper.scalar_multiply_py_matrix(diff, 2 / (y_true.cols * y_true.rows))
        c_wrapper.free_py_matrix(diff)
        
        num_layers = len(self.weights)

        # Backpropagate through layers
        for layer_index in range(num_layers - 1, -1, -1):
            
            is_output_layer = (layer_index == num_layers - 1)
            activation_type = self.output_activation if is_output_layer else self.hidden_activation
            
            if activation_type == "sigmoid":
                activation_derivative = c_wrapper.py_sigmoid_derivative(self.activations[layer_index + 1])
            elif activation_type == "relu":
                activation_derivative = c_wrapper.py_matrix_relu_derivative(self.activations[layer_index + 1])
            elif activation_type == "identity":
                activation_derivative = None
                pass
            else:
                raise ValueError(f"Unsupported activation {activation_type} in backward_pass")
            
            # Compute error signal (delta)
            if layer_index == num_layers - 1:
                # Output Layer: Loss_grad * Act_Derivative
                if activation_type == "identity":
                    # Derivative of 1: Error passes straight through
                    error_signal = output_error 
                else:
                    error_signal = c_wrapper.hadamard_py_matrices(output_error, activation_derivative)
            else:
                # Hidden Layer: (W_T * Previous_Error) * Act_Derivative
                transposed_weights = c_wrapper.transpose_py_matrix(self.weights[layer_index + 1])
                backpropagate = c_wrapper.multiply_py_matrices(transposed_weights, output_error)
                c_wrapper.free_py_matrix(transposed_weights)

                if activation_type == "identity":
                    error_signal = backpropagate
                else:
                    error_signal = c_wrapper.hadamard_py_matrices(backpropagate, activation_derivative)
                
                c_wrapper.free_py_matrix(backpropagate)
                
            # Weight Gradient = Error_Signal * A_{L-1}^T
            transposed_prev = c_wrapper.transpose_py_matrix(self.activations[layer_index])
            weight_grad = c_wrapper.scalar_multiply_py_matrix(c_wrapper.multiply_py_matrices(error_signal, transposed_prev), self.learning_rate)
            # Bias Gradient = Sum of error signal columns
            bias_grad = c_wrapper.scalar_multiply_py_matrix(c_wrapper.sum_py_matrix_columns(error_signal), self.learning_rate)
            
            # W_new = W_old - Grad
            new_weights = c_wrapper.subtract_py_matrices(self.weights[layer_index], weight_grad)
            new_biases = c_wrapper.subtract_py_matrices(self.biases[layer_index], bias_grad)

            c_wrapper.free_py_matrix(self.weights[layer_index])
            c_wrapper.free_py_matrix(self.biases[layer_index])

            self.weights[layer_index] = new_weights
            self.biases[layer_index] = new_biases

            if activation_derivative is not None:
                c_wrapper.free_py_matrix(activation_derivative)
            c_wrapper.free_py_matrix(transposed_prev)
            c_wrapper.free_py_matrix(weight_grad)
            c_wrapper.free_py_matrix(bias_grad)
                
            if output_error.c_ptr != error_signal.c_ptr:
                c_wrapper.free_py_matrix(output_error)
            output_error = error_signal
        
        c_wrapper.free_py_matrix(output_error)

    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int):
        X_c = c_wrapper.transpose_py_matrix(c_wrapper.from_numpy(X))
        y_c = c_wrapper.transpose_py_matrix(c_wrapper.from_numpy(y))

        for epoch in range(epochs):
            print(f"Training epoch {epoch}\n")

            y_pred = self.forward_pass(X_c)
            
            loss = c_wrapper.py_mean_squared_error(y_c, y_pred)
            print(f"This epoch's loss is {loss:.6f}")

            self.backward_pass(y_c, y_pred)

            self._clear_activations()

        c_wrapper.free_py_matrix(X_c)
        c_wrapper.free_py_matrix(y_c)

        print("Training Complete")

    def predict(self, X: np.ndarray):
        X_c = c_wrapper.transpose_py_matrix(c_wrapper.from_numpy(X))

        y_pred = self.forward_pass(X_c)
        # Transpose the output back to NumPy's (samples, features) format
        y_pred_transposed = c_wrapper.transpose_py_matrix(y_pred)

        prediction = c_wrapper.to_numpy(y_pred_transposed)

        c_wrapper.free_py_matrix(X_c)
        c_wrapper.free_py_matrix(y_pred_transposed)
        self._clear_activations()

        return prediction

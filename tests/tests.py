import sys
import os
import random
import numpy as np
import traceback

# Add the parent directory to the path to import mlp and c_wrapper
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))
from mlp import MLP
from c_wrapper import (
    py_list_to_c_matrix,
    c_matrix_to_py_list,
    add_py_matrices,
    subtract_py_matrices,
    multiply_py_matrices,
    transpose_py_matrix,
    scalar_multiply_py_matrix,
    free_py_matrix,
    create_zero_py_matrix,
    from_numpy,
    to_numpy,
    py_sigmoid,
    py_sigmoid_derivative,
)

def test_matrix_operations():
    """Test basic matrix operations"""
    print("=" * 50)
    print("TESTING MATRIX OPERATIONS")
    print("=" * 50)
    
    py_mat1 = [[1.2, 2.99], [3.1, 4.25]]
    py_mat2 = [[5.01, 6.91], [7.25, 8.765]]
    
    py_sigmoid_test_mat = [[-1.0, 0.0, 3.0], [5.0, -2.5, 1.5]]

    c_mat1 = None
    c_mat2 = None
    c_result = None
    c_zero_mat = None
    c_sum_mat = None
    c_sigmoid_test_mat = None
    c_sigmoid_result = None
    c_sigmoid_deriv_result = None
    
    try:
        c_mat1 = py_list_to_c_matrix(py_mat1)
        c_mat2 = py_list_to_c_matrix(py_mat2)
        
        c_result = multiply_py_matrices(c_mat1, c_mat2)
        py_result = c_matrix_to_py_list(c_result)
        
        print("Matrix 1:")
        print(py_mat1)
        print("\nMatrix 2:")
        print(py_mat2)
        print("\nResult of matrix multiplication:")
        print(py_result)
        
        c_zero_mat = create_zero_py_matrix(3, 3)
        print("\nZero Matrix:")
        print(c_matrix_to_py_list(c_zero_mat))
        
        c_sum_mat = add_py_matrices(c_mat1, c_mat2)
        print("\nResult of matrix addition:")
        print(c_matrix_to_py_list(c_sum_mat))
        
        # --- NEW SIGMOID TESTS ---
        print("\n" + "-" * 20)
        print("TESTING MATRIX-WISE SIGMOID FUNCTIONS")
        print("-" * 20)
        
        c_sigmoid_test_mat = py_list_to_c_matrix(py_sigmoid_test_mat)

        c_sigmoid_result = py_sigmoid(c_sigmoid_test_mat)
        print("\nInput matrix for sigmoid:")
        print(py_sigmoid_test_mat)
        print("\nResult of matrix_sigmoid:")
        print(c_matrix_to_py_list(c_sigmoid_result))

        expected_sigmoid = 1 / (1 + np.exp(-np.array(py_sigmoid_test_mat)))
        print("\nExpected result (from numpy):")
        print(expected_sigmoid.tolist())
        
        c_sigmoid_deriv_result = py_sigmoid_derivative(c_sigmoid_test_mat)
        print("\nResult of matrix_sigmoid_derivative:")
        print(c_matrix_to_py_list(c_sigmoid_deriv_result))

        expected_sigmoid_deriv = expected_sigmoid * (1 - expected_sigmoid)
        print("\nExpected derivative result (from numpy):")
        print(expected_sigmoid_deriv.tolist())
        
    finally:
        for mat in [
            c_mat1, c_mat2, c_result, c_zero_mat, c_sum_mat,
            c_sigmoid_test_mat, c_sigmoid_result, c_sigmoid_deriv_result
        ]:
            if mat:
                free_py_matrix(mat)

def test_mlp_initialization():
    """Test MLP initialization with various configurations"""
    print("\n" + "=" * 50)
    print("TESTING MLP INITIALIZATION")
    print("=" * 50)
    
    print("\nTest 1: Valid MLP initialization")
    try:
        mlp = MLP([2, 3, 1], activation="Sigmoid", loss="MSE", learning_rate=0.01, seed=42)
        print(f"✓ MLP created successfully with layer sizes: {mlp.layer_sizes}")
        del mlp
        print("✓ MLP destructor called successfully")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\nTest 2: MLP with multiple hidden layers")
    try:
        mlp = MLP([4, 8, 6, 3, 1], seed=123)
        print(f"✓ Multi-layer MLP created: {mlp.layer_sizes}")
        del mlp
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\nTest 3: Invalid layer sizes (< 2 layers)")
    try:
        mlp = MLP([5], seed=456)
        print("✗ This should have printed an error message")
        del mlp
    except Exception as e:
        print(f"Expected behavior - Error caught: {e}")
    
    print("\nTest 4: Minimum valid layer configuration")
    try:
        mlp = MLP([1, 1], seed=789)
        print("✓ Minimum MLP created successfully")
        del mlp
    except Exception as e:
        print(f"✗ Error: {e}")

def test_mlp_weight_initialization():
    """Test that weights are properly initialized with Xavier/Glorot initialization"""
    print("\n" + "=" * 50)
    print("TESTING MLP WEIGHT INITIALIZATION")
    print("=" * 50)
    
    np.random.seed(42)
    mlp = MLP([3, 4, 2], seed=42)
    
    print("Checking weight initialization bounds (Xavier/Glorot):")
    
    for i, (w, b) in enumerate(zip(mlp.weights, mlp.biases)):
        print(f"\nLayer {i} -> {i+1}:")
        w_np = to_numpy(w)
        b_np = to_numpy(b)
        
        print(f"✓ Weight matrix initialized: {w_np.shape}")
        print(f"✓ Bias vector initialized: {b_np.shape}")
        
        layer_in = mlp.layer_sizes[i]
        layer_out = mlp.layer_sizes[i+1]
        expected_limit = np.sqrt(6 / (layer_in + layer_out))
        
        is_in_range = np.all(np.abs(w_np) <= expected_limit)
        print(f"✓ Weights are within expected range ±{expected_limit:.4f}: {is_in_range}")

def test_mlp_memory_management():
    """Test memory management and multiple MLPs"""
    print("\n" + "=" * 50)
    print("TESTING MLP MEMORY MANAGEMENT")
    print("=" * 50)
    
    print("Creating and destroying multiple MLPs...")
    
    mlps = []
    try:
        for i in range(5):
            mlp = MLP([2, 4, 2], seed=i)
            mlps.append(mlp)
            print(f"✓ MLP {i+1} created")
        
        print("✓ All MLPs created successfully")
        
        for i, mlp in enumerate(mlps):
            del mlp
            print(f"✓ MLP {i+1} destroyed")
            
    except Exception as e:
        print(f"✗ Error in memory management test: {e}")

def test_mlp_with_xor_data():
    """Test MLP setup with XOR training data"""
    print("\n" + "=" * 50)
    print("TESTING MLP WITH XOR DATA SETUP")
    print("=" * 50)
    
    training_data = [
        ([0, 0], [0]), ([0, 1], [1]),
        ([1, 0], [1]), ([1, 1], [0])
    ]
    
    print("XOR Training Data:")
    for inputs, targets in training_data:
        print(f"  Input: {inputs} -> Target: {targets}")
    
    print("\nCreating MLP for XOR problem...")
    try:
        mlp = MLP([2, 4, 1], activation="Sigmoid", loss="MSE", learning_rate=0.1, seed=42)
        print("✓ XOR MLP created successfully")
        print(f"  Architecture: {mlp.layer_sizes}")
        
        print("\nTesting data format compatibility...")
        for i, (inputs, targets) in enumerate(training_data):
            inputs_matrix = from_numpy(np.array(inputs, dtype=np.float64).reshape(-1, 1))
            free_py_matrix(inputs_matrix)
            print(f"  Sample {i+1}: {inputs} -> {targets} ✓")
        
        del mlp
        print("✓ XOR MLP test completed")
        
    except Exception as e:
        print(f"✗ Error in XOR MLP test: {e}")

def test_single_layer_forward():
    """Test the forward pass between two specific layers"""
    print("\n" + "=" * 50)
    print("TESTING SINGLE LAYER FORWARD PASS")
    print("=" * 50)
    
    print("\nTest 1: Forward pass from input to first hidden layer")
    try:
        mlp = MLP([2, 3, 1], activation="Sigmoid", seed=42)
        test_input_data = [[0.5], [0.8]]
        input_matrix = py_list_to_c_matrix(test_input_data)
        
        layer_0_output = mlp.forward_pass_one_layer(0, input_matrix)
        result = c_matrix_to_py_list(layer_0_output)
        
        print(f"✓ Input shape: {len(test_input_data)}x{len(test_input_data[0])}")
        print(f"✓ Output shape: {len(result)}x{len(result[0])}")
        print(f"✓ Expected output shape: 3x1 (hidden layer size)")
        
        all_valid = all(0 <= row[0] <= 1 for row in result)
        print(f"✓ All outputs in sigmoid range [0,1]: {all_valid}")
        
        free_py_matrix(input_matrix)
        free_py_matrix(layer_0_output)
        del mlp
        
    except Exception as e:
        print(f"✗ Error in first layer test: {e}")
        traceback.print_exc()
    
    print("\nTest 2: Forward pass from hidden to output layer")
    try:
        mlp = MLP([2, 3, 1], activation="Sigmoid", seed=123)
        hidden_input_data = [[0.6], [0.4], [0.7]]
        hidden_matrix = py_list_to_c_matrix(hidden_input_data)
        
        layer_1_output = mlp.forward_pass_one_layer(1, hidden_matrix)
        result = c_matrix_to_py_list(layer_1_output)
        
        print(f"✓ Hidden input shape: {len(hidden_input_data)}x{len(hidden_input_data[0])}")
        print(f"✓ Output shape: {len(result)}x{len(result[0])}")
        print(f"✓ Expected output shape: 1x1 (output layer size)")
        
        output_valid = 0 <= result[0][0] <= 1
        print(f"✓ Output in sigmoid range [0,1]: {output_valid}")
        
        free_py_matrix(hidden_matrix)
        free_py_matrix(layer_1_output)
        del mlp
        
    except Exception as e:
        print(f"✗ Error in second layer test: {e}")

def test_layer_forward_chain():
    """Test chaining layer forward passes together"""
    print("\n" + "=" * 50)
    print("TESTING CHAINED LAYER FORWARD PASSES")
    print("=" * 50)
    
    try:
        mlp = MLP([2, 4, 3, 1], activation="Sigmoid", seed=456)
        input_data = [[0.3], [0.9]]
        current_matrix = py_list_to_c_matrix(input_data)
        
        print(f"Initial input: {[row[0] for row in input_data]}")
        
        for layer_idx in range(len(mlp.layer_sizes) - 1):
            next_matrix = mlp.forward_pass_one_layer(layer_idx, current_matrix)
            result = c_matrix_to_py_list(next_matrix)
            
            print(f"✓ Layer {layer_idx} -> {layer_idx+1}: {[row[0] for row in result]}")
            
            if layer_idx > 0:
                free_py_matrix(current_matrix)
            
            current_matrix = next_matrix
        
        final_result = c_matrix_to_py_list(current_matrix)
        print(f"\n✓ Final network output: {final_result[0][0]:.6f}")
        print("✓ Complete forward pass successful!")
        
        free_py_matrix(current_matrix)
        del mlp
        
    except Exception as e:
        print(f"✗ Error in chained forward pass: {e}")
        traceback.print_exc()

def test_layer_forward_error_cases():
    """Test error handling in layer forward pass"""
    print("\n" + "=" * 50)
    print("TESTING LAYER FORWARD ERROR CASES")
    print("=" * 50)
    
    print("\nTest 1: Invalid layer index")
    try:
        mlp = MLP([2, 3, 1], activation="Sigmoid", seed=789)
        input_data = [[0.5], [0.5]]
        input_matrix = py_list_to_c_matrix(input_data)
        
        try:
            mlp.forward_pass_one_layer(2, input_matrix)
            print("✗ Should have failed with invalid layer index")
        except IndexError:
            print("✓ Correctly caught invalid layer index.")
        
        free_py_matrix(input_matrix)
        del mlp
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    print("\nTest 2: Wrong input matrix dimensions")
    try:
        mlp = MLP([2, 3, 1], activation="Sigmoid", seed=101112)
        wrong_input_data = [[0.1], [0.2], [0.3]]
        wrong_matrix = py_list_to_c_matrix(wrong_input_data)
        
        try:
            mlp.forward_pass_one_layer(0, wrong_matrix)
            print("✗ Should have failed with dimension mismatch")
        except ValueError:
            print("✓ Correctly caught dimension error.")
        
        free_py_matrix(wrong_matrix)
        del mlp
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    print("\nTest 3: Unsupported activation function")
    try:
        mlp = MLP([2, 2, 1], activation="ReLU", seed=131415)
        input_data = [[0.5], [0.5]]
        input_matrix = py_list_to_c_matrix(input_data)
        
        try:
            mlp.forward_pass_one_layer(0, input_matrix)
            print("✗ Should have failed with unsupported activation")
        except NotImplementedError:
            print("✓ Correctly caught activation error.")
        
        free_py_matrix(input_matrix)
        del mlp
        
    except Exception as e:
        print(f"Note: {e} (expected if ReLU not supported in constructor)")

def test_layer_forward_mathematical_correctness():
    """Test that the mathematical computation is correct"""
    print("\n" + "=" * 50)
    print("TESTING MATHEMATICAL CORRECTNESS")
    print("=" * 50)
    
    try:
        np.random.seed(42)
        mlp = MLP([2, 1], activation="Sigmoid", seed=42)
        input_data = [[1.0], [0.5]]
        input_matrix = py_list_to_c_matrix(input_data)
        
        weights_np = to_numpy(mlp.weights[0])
        biases_np = to_numpy(mlp.biases[0])
        input_np = np.array(input_data)

        linear_input_np = np.dot(weights_np, input_np) + biases_np
        expected_output_np = 1 / (1 + np.exp(-linear_input_np))

        result_matrix = mlp.forward_pass_one_layer(0, input_matrix)
        result = to_numpy(result_matrix)
        network_output = result[0][0]
        
        print(f"✓ Network computed output: {network_output:.6f}")
        print(f"✓ Expected output (manual calculation): {expected_output_np[0][0]:.6f}")

        if np.isclose(network_output, expected_output_np[0][0]):
            print("✓ Outputs are mathematically correct!")
        else:
            print("✗ Outputs do not match expected value.")
        
        free_py_matrix(input_matrix)
        free_py_matrix(result_matrix)
        del mlp
        
    except Exception as e:
        print(f"✗ Error in mathematical correctness test: {e}")
        traceback.print_exc()

def run_layer_forward_tests():
    """Run all layer forward pass tests"""
    print("\nSINGLE LAYER FORWARD PASS TESTING SUITE")
    print("=" * 60)
    
    try:
        test_single_layer_forward()
        test_layer_forward_chain()
        test_layer_forward_error_cases()
        test_layer_forward_mathematical_correctness()
        
        print("\n" + "=" * 60)
        print("ALL LAYER FORWARD TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR IN LAYER FORWARD TEST SUITE: {e}")
        traceback.print_exc()

def run_all_tests():
    """Run all MLP tests"""
    print("MLP NEURAL NETWORK TESTING SUITE")
    print("=" * 60)
    
    try:
        test_matrix_operations()
        
        test_mlp_initialization()
        test_mlp_weight_initialization()
        test_mlp_memory_management()
        test_mlp_with_xor_data()
        test_mlp_different_configurations()
        run_layer_forward_tests()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR IN TEST SUITE: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()

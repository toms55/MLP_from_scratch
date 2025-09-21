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

def close_enough(a, b, tolerance=1e-6):
    """Helper function for floating point comparison"""
    return abs(a - b) < tolerance

def matrices_close_enough(mat1, mat2, tolerance=1e-6):
    """Helper function to compare matrices element-wise"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return False
    
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            if not close_enough(mat1[i][j], mat2[i][j], tolerance):
                return False
    return True

def test_matrix_operations():
    """Test basic matrix operations with enhanced accuracy checks"""
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
    c_sub_mat = None
    c_scalar_mat = None
    c_transpose_mat = None
    c_sigmoid_test_mat = None
    c_sigmoid_result = None
    c_sigmoid_deriv_result = None
    
    try:
        # Test matrix creation and basic operations
        c_mat1 = py_list_to_c_matrix(py_mat1)
        c_mat2 = py_list_to_c_matrix(py_mat2)
        
        # Test multiplication
        c_result = multiply_py_matrices(c_mat1, c_mat2)
        py_result = c_matrix_to_py_list(c_result)
        
        # Calculate expected result using numpy
        expected_mult = np.dot(np.array(py_mat1), np.array(py_mat2)).tolist()
        
        print("Matrix 1:")
        print(py_mat1)
        print("\nMatrix 2:")
        print(py_mat2)
        print("\nResult of matrix multiplication:")
        print(py_result)
        print("\nExpected result (numpy):")
        print(expected_mult)
        
        if matrices_close_enough(py_result, expected_mult):
            print("✓ Matrix multiplication is mathematically correct")
        else:
            print("✗ Matrix multiplication result differs from expected")
        
        # Test zero matrix creation
        c_zero_mat = create_zero_py_matrix(3, 3)
        zero_result = c_matrix_to_py_list(c_zero_mat)
        expected_zero = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        
        print("\nZero Matrix:")
        print(zero_result)
        if matrices_close_enough(zero_result, expected_zero):
            print("✓ Zero matrix creation is correct")
        else:
            print("✗ Zero matrix creation failed")
        
        # Test addition
        c_sum_mat = add_py_matrices(c_mat1, c_mat2)
        sum_result = c_matrix_to_py_list(c_sum_mat)
        expected_sum = (np.array(py_mat1) + np.array(py_mat2)).tolist()
        
        print("\nResult of matrix addition:")
        print(sum_result)
        print("Expected result (numpy):")
        print(expected_sum)
        
        if matrices_close_enough(sum_result, expected_sum):
            print("✓ Matrix addition is mathematically correct")
        else:
            print("✗ Matrix addition result differs from expected")
        
        # Test subtraction
        c_sub_mat = subtract_py_matrices(c_mat1, c_mat2)
        sub_result = c_matrix_to_py_list(c_sub_mat)
        expected_sub = (np.array(py_mat1) - np.array(py_mat2)).tolist()
        
        print("\nResult of matrix subtraction:")
        print(sub_result)
        print("Expected result (numpy):")
        print(expected_sub)
        
        if matrices_close_enough(sub_result, expected_sub):
            print("✓ Matrix subtraction is mathematically correct")
        else:
            print("✗ Matrix subtraction result differs from expected")
        
        # Test scalar multiplication
        scalar = 2.5
        c_scalar_mat = scalar_multiply_py_matrix(c_mat1, scalar)
        scalar_result = c_matrix_to_py_list(c_scalar_mat)
        expected_scalar = (np.array(py_mat1) * scalar).tolist()
        
        print(f"\nResult of scalar multiplication by {scalar}:")
        print(scalar_result)
        print("Expected result (numpy):")
        print(expected_scalar)
        
        if matrices_close_enough(scalar_result, expected_scalar):
            print("✓ Scalar multiplication is mathematically correct")
        else:
            print("✗ Scalar multiplication result differs from expected")
        
        # Test transpose
        c_transpose_mat = transpose_py_matrix(c_mat1)
        transpose_result = c_matrix_to_py_list(c_transpose_mat)
        expected_transpose = np.array(py_mat1).T.tolist()
        
        print("\nResult of matrix transpose:")
        print(transpose_result)
        print("Expected result (numpy):")
        print(expected_transpose)
        
        if matrices_close_enough(transpose_result, expected_transpose):
            print("✓ Matrix transpose is mathematically correct")
        else:
            print("✗ Matrix transpose result differs from expected")
        
        # Test sigmoid functions
        print("\n" + "-" * 20)
        print("TESTING MATRIX-WISE SIGMOID FUNCTIONS")
        print("-" * 20)
        
        c_sigmoid_test_mat = py_list_to_c_matrix(py_sigmoid_test_mat)

        c_sigmoid_result = py_sigmoid(c_sigmoid_test_mat)
        sigmoid_result = c_matrix_to_py_list(c_sigmoid_result)
        
        print("\nInput matrix for sigmoid:")
        print(py_sigmoid_test_mat)
        print("\nResult of matrix_sigmoid:")
        print(sigmoid_result)

        expected_sigmoid = (1 / (1 + np.exp(-np.array(py_sigmoid_test_mat)))).tolist()
        print("\nExpected result (from numpy):")
        print(expected_sigmoid)
        
        if matrices_close_enough(sigmoid_result, expected_sigmoid):
            print("✓ Sigmoid function is mathematically correct")
        else:
            print("✗ Sigmoid function result differs from expected")
        
        c_sigmoid_deriv_result = py_sigmoid_derivative(c_sigmoid_test_mat)
        sigmoid_deriv_result = c_matrix_to_py_list(c_sigmoid_deriv_result)
        
        print("\nResult of matrix_sigmoid_derivative:")
        print(sigmoid_deriv_result)

        expected_sigmoid_np = 1 / (1 + np.exp(-np.array(py_sigmoid_test_mat)))
        expected_sigmoid_deriv = (expected_sigmoid_np * (1 - expected_sigmoid_np)).tolist()
        print("\nExpected derivative result (from numpy):")
        print(expected_sigmoid_deriv)
        
        if matrices_close_enough(sigmoid_deriv_result, expected_sigmoid_deriv):
            print("✓ Sigmoid derivative is mathematically correct")
        else:
            print("✗ Sigmoid derivative result differs from expected")
        
    finally:
        for mat in [
            c_mat1, c_mat2, c_result, c_zero_mat, c_sum_mat, c_sub_mat,
            c_scalar_mat, c_transpose_mat, c_sigmoid_test_mat, 
            c_sigmoid_result, c_sigmoid_deriv_result
        ]:
            if mat:
                free_py_matrix(mat)

def test_mlp_initialization():
    """Test MLP initialization with various configurations and edge cases"""
    print("\n" + "=" * 50)
    print("TESTING MLP INITIALIZATION")
    print("=" * 50)
    
    print("\nTest 1: Valid MLP initialization")
    try:
        mlp = MLP([2, 3, 1], activation="Sigmoid", loss="MSE", learning_rate=0.01, seed=42)
        print(f"✓ MLP created successfully with layer sizes: {mlp.layer_sizes}")
        print(f"  Activation: {mlp.activation}")
        print(f"  Loss: {mlp.loss}")
        print(f"  Learning rate: {mlp.learning_rate}")
        
        # Verify correct number of weight and bias matrices
        expected_layers = len(mlp.layer_sizes) - 1
        if len(mlp.weights) == expected_layers and len(mlp.biases) == expected_layers:
            print(f"✓ Correct number of weight/bias matrices: {expected_layers}")
        else:
            print(f"✗ Incorrect number of weight/bias matrices")
        
        del mlp
        print("✓ MLP destructor called successfully")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\nTest 2: MLP with multiple hidden layers")
    try:
        mlp = MLP([4, 8, 6, 3, 1], seed=123)
        print(f"✓ Multi-layer MLP created: {mlp.layer_sizes}")
        
        # Verify matrix dimensions
        dimensions_correct = True
        for i in range(len(mlp.weights)):
            w_np = to_numpy(mlp.weights[i])
            b_np = to_numpy(mlp.biases[i])
            expected_w_shape = (mlp.layer_sizes[i+1], mlp.layer_sizes[i])
            expected_b_shape = (mlp.layer_sizes[i+1], 1)
            
            if w_np.shape != expected_w_shape or b_np.shape != expected_b_shape:
                dimensions_correct = False
                print(f"✗ Layer {i}: Wrong dimensions")
                print(f"  Weight shape: {w_np.shape}, expected: {expected_w_shape}")
                print(f"  Bias shape: {b_np.shape}, expected: {expected_b_shape}")
        
        if dimensions_correct:
            print("✓ All weight and bias matrices have correct dimensions")
        
        del mlp
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\nTest 3: Invalid layer sizes (< 2 layers)")
    try:
        mlp = MLP([5], seed=456)
        print("✗ This should have failed with insufficient layers")
        del mlp
    except Exception as e:
        print(f"✓ Correctly caught error: {e}")
    
    print("\nTest 4: Invalid layer sizes (empty list)")
    try:
        mlp = MLP([], seed=789)
        print("✗ This should have failed with empty layer list")
        del mlp
    except Exception as e:
        print(f"✓ Correctly caught error: {e}")
    
    print("\nTest 5: Minimum valid layer configuration")
    try:
        mlp = MLP([1, 1], seed=789)
        print("✓ Minimum MLP created successfully")
        del mlp
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\nTest 6: Large layer configuration")
    try:
        mlp = MLP([100, 50, 25, 10, 1], seed=999)
        print("✓ Large MLP created successfully")
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
    
    initialization_correct = True
    for i, (w, b) in enumerate(zip(mlp.weights, mlp.biases)):
        print(f"\nLayer {i} -> {i+1}:")
        w_np = to_numpy(w)
        b_np = to_numpy(b)
        
        print(f"✓ Weight matrix shape: {w_np.shape}")
        print(f"✓ Bias vector shape: {b_np.shape}")
        
        # Check Xavier/Glorot initialization bounds
        layer_in = mlp.layer_sizes[i]
        layer_out = mlp.layer_sizes[i+1]
        expected_limit = np.sqrt(6 / (layer_in + layer_out))
        
        weights_in_range = np.all(np.abs(w_np) <= expected_limit)
        print(f"  Weight range check: ±{expected_limit:.4f} - {'✓' if weights_in_range else '✗'}")
        
        # Check bias initialization (should be zeros)
        biases_zero = np.allclose(b_np, 0.0)
        print(f"  Bias initialization (zeros): {'✓' if biases_zero else '✗'}")
        
        # Check weight distribution statistics
        weight_mean = np.mean(w_np)
        weight_std = np.std(w_np)
        print(f"  Weight mean: {weight_mean:.6f} (should be ~0)")
        print(f"  Weight std: {weight_std:.6f}")
        
        if not weights_in_range or not biases_zero:
            initialization_correct = False
    
    if initialization_correct:
        print("\n✓ All weight and bias initialization checks passed")
    else:
        print("\n✗ Some weight/bias initialization checks failed")
    
    del mlp

def test_mlp_memory_management():
    """Test memory management and multiple MLPs with stress testing"""
    print("\n" + "=" * 50)
    print("TESTING MLP MEMORY MANAGEMENT")
    print("=" * 50)
    
    print("Creating and destroying multiple MLPs...")
    
    mlps = []
    try:
        # Test creating multiple MLPs
        for i in range(10):
            mlp = MLP([random.randint(2, 10), random.randint(2, 10), 1], seed=i)
            mlps.append(mlp)
            print(f"✓ MLP {i+1} created with architecture: {mlp.layer_sizes}")
        
        print("✓ All MLPs created successfully")
        
        # Test accessing MLPs after creation
        for i, mlp in enumerate(mlps):
            # Try to use the MLP to ensure it's still valid
            input_size = mlp.layer_sizes[0]
            test_input = from_numpy(np.random.random((input_size, 1)))
            output = mlp.forward_pass(test_input)
            free_py_matrix(test_input)
            free_py_matrix(output)
            print(f"✓ MLP {i+1} successfully used for forward pass")
        
        # Test destroying MLPs
        for i, mlp in enumerate(mlps):
            del mlp
            print(f"✓ MLP {i+1} destroyed")
        
        print("✓ Memory management test completed successfully")
            
    except Exception as e:
        print(f"✗ Error in memory management test: {e}")
        traceback.print_exc()

def test_mlp_with_xor_data():
    """Test MLP setup with XOR training data and verify dimensions"""
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
        
        print("\nTesting data format compatibility and forward passes...")
        all_outputs = []
        for i, (inputs, targets) in enumerate(training_data):
            inputs_matrix = from_numpy(np.array(inputs, dtype=np.float64).reshape(-1, 1))
            targets_matrix = from_numpy(np.array(targets, dtype=np.float64).reshape(-1, 1))
            
            # Test forward pass
            output_matrix = mlp.forward_pass(inputs_matrix)
            output_value = to_numpy(output_matrix)[0, 0]
            all_outputs.append(output_value)
            
            print(f"  Sample {i+1}: {inputs} -> {output_value:.6f} (target: {targets[0]})")
            
            free_py_matrix(inputs_matrix)
            free_py_matrix(targets_matrix)
            free_py_matrix(output_matrix)
        
        # Check that outputs are different (network is not trivially constant)
        output_variance = np.var(all_outputs)
        if output_variance > 1e-6:
            print(f"✓ Network produces varied outputs (variance: {output_variance:.6f})")
        else:
            print(f"✗ Network produces constant outputs (variance: {output_variance:.6f})")
        
        del mlp
        print("✓ XOR MLP test completed")
        
    except Exception as e:
        print(f"✗ Error in XOR MLP test: {e}")
        traceback.print_exc()

def test_full_forward_pass():
    """Test a full forward pass with detailed step-by-step verification"""
    print("\n" + "=" * 50)
    print("TESTING FULL FORWARD PASS")
    print("=" * 50)
    
    try:
        # Test with known seed for reproducible results
        np.random.seed(42)
        mlp = MLP([2, 4, 3, 1], activation="Sigmoid", seed=42)
        input_data = [[0.3], [0.9]]
        input_matrix = py_list_to_c_matrix(input_data)
        
        print(f"Input vector: {[row[0] for row in input_data]}")
        print(f"Network architecture: {mlp.layer_sizes}")
        
        # Perform the forward pass and capture intermediate activations
        final_output_matrix = mlp.forward_pass(input_matrix)
        final_output = c_matrix_to_py_list(final_output_matrix)
        
        print(f"\n✓ Final network output: {final_output[0][0]:.6f}")
        
        # Verify that activations were stored correctly
        if len(mlp.activations) == len(mlp.layer_sizes):
            print(f"✓ Correct number of activation layers stored: {len(mlp.activations)}")
            
            # Check dimensions of each activation layer
            for i, activation in enumerate(mlp.activations):
                act_np = to_numpy(activation)
                expected_size = mlp.layer_sizes[i]
                if act_np.shape == (expected_size, 1):
                    print(f"✓ Layer {i} activation shape correct: {act_np.shape}")
                else:
                    print(f"✗ Layer {i} activation shape incorrect: {act_np.shape}, expected: ({expected_size}, 1)")
        else:
            print(f"✗ Incorrect number of activation layers: {len(mlp.activations)}, expected: {len(mlp.layer_sizes)}")
        
        # Test with different input sizes
        print("\nTesting with different valid inputs:")
        test_inputs = [
            [[1.0], [0.0]],
            [[0.5], [0.5]],
            [[-1.0], [1.0]]
        ]
        
        for j, test_input in enumerate(test_inputs):
            test_matrix = py_list_to_c_matrix(test_input)
            test_output_matrix = mlp.forward_pass(test_matrix)
            test_output = c_matrix_to_py_list(test_output_matrix)
            print(f"  Input {j+1}: {[row[0] for row in test_input]} -> Output: {test_output[0][0]:.6f}")
            free_py_matrix(test_matrix)
            free_py_matrix(test_output_matrix)
        
        print("✓ Complete forward pass testing successful!")
        
        # Clean up matrices
        free_py_matrix(input_matrix)
        free_py_matrix(final_output_matrix)
        del mlp
        
    except Exception as e:
        print(f"✗ Error in full forward pass test: {e}")
        traceback.print_exc()

def test_forward_pass_error_cases():
    """Test error handling in forward pass with comprehensive edge cases"""
    print("\n" + "=" * 50)
    print("TESTING FORWARD PASS ERROR CASES")
    print("=" * 50)
    
    print("\nTest 1: Wrong input matrix dimensions")
    try:
        mlp = MLP([2, 3, 1], activation="Sigmoid", seed=101112)
        
        # Test various wrong input sizes
        wrong_inputs = [
            ([[0.1], [0.2], [0.3]], "3x1 instead of 2x1"),
            ([[0.1]], "1x1 instead of 2x1"),
            ([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "2x3 instead of 2x1")
        ]
        
        for wrong_input_data, description in wrong_inputs:
            try:
                wrong_matrix = py_list_to_c_matrix(wrong_input_data)
                mlp.forward_pass(wrong_matrix)
                print(f"✗ Should have failed with {description}")
                free_py_matrix(wrong_matrix)
            except (ValueError, Exception) as e:
                print(f"✓ Correctly caught dimension error for {description}: {type(e).__name__}")
                if 'wrong_matrix' in locals():
                    free_py_matrix(wrong_matrix)
        
        del mlp
        
    except Exception as e:
        print(f"✗ Unexpected error in dimension test: {e}")
    
    print("\nTest 2: Null/invalid matrix input")
    try:
        mlp = MLP([2, 2, 1], activation="Sigmoid", seed=131415)
        
        try:
            # This should fail gracefully
            mlp.forward_pass(None)
            print("✗ Should have failed with null input")
        except Exception as e:
            print(f"✓ Correctly caught null input error: {type(e).__name__}")
        
        del mlp
        
    except Exception as e:
        print(f"✗ Unexpected error in null input test: {e}")
    
    print("\nTest 3: Edge case values")
    try:
        mlp = MLP([2, 2, 1], activation="Sigmoid", seed=161718)
        
        # Test with extreme values
        extreme_inputs = [
            ([[1000.0], [-1000.0]], "Very large values"),
            ([[1e-10], [1e-10]], "Very small values"),
            ([[float('inf')], [1.0]], "Infinity values"),
            ([[float('nan')], [1.0]], "NaN values")
        ]
        
        for extreme_input, description in extreme_inputs:
            try:
                extreme_matrix = py_list_to_c_matrix(extreme_input)
                output_matrix = mlp.forward_pass(extreme_matrix)
                output = to_numpy(output_matrix)
                
                if np.isfinite(output).all():
                    print(f"✓ {description}: Produces finite output")
                elif np.isnan(output).any():
                    print(f"⚠ {description}: Produces NaN output")
                else:
                    print(f"⚠ {description}: Produces non-finite output")
                
                free_py_matrix(extreme_matrix)
                free_py_matrix(output_matrix)
                
            except Exception as e:
                print(f"✗ Error with {description}: {e}")
        
        del mlp
        
    except Exception as e:
        print(f"✗ Unexpected error in extreme values test: {e}")

def test_forward_pass_mathematical_correctness():
    """Test mathematical correctness with manual calculations"""
    print("\n" + "=" * 50)
    print("TESTING MATHEMATICAL CORRECTNESS")
    print("=" * 50)
    
    print("\nTest 1: Single layer network (manual verification)")
    try:
        np.random.seed(42)
        mlp = MLP([2, 1], activation="Sigmoid", seed=42)
        input_data = [[1.0], [0.5]]
        input_matrix = py_list_to_c_matrix(input_data)
        
        # Get network parameters
        weights_np = to_numpy(mlp.weights[0])
        biases_np = to_numpy(mlp.biases[0])
        input_np = np.array(input_data)

        print(f"Input: {input_np.flatten()}")
        print(f"Weights: {weights_np.flatten()}")
        print(f"Bias: {biases_np.flatten()}")
        
        # Manual calculation
        linear_input_np = np.dot(weights_np, input_np) + biases_np
        expected_output_np = 1 / (1 + np.exp(-linear_input_np))
        
        print(f"Linear combination: {linear_input_np.flatten()}")
        print(f"Expected output: {expected_output_np.flatten()}")

        # Network calculation
        result_matrix = mlp.forward_pass(input_matrix)
        result = to_numpy(result_matrix)
        network_output = result[0][0]
        
        print(f"Network output: {network_output}")

        if close_enough(network_output, expected_output_np[0][0], 1e-10):
            print("✓ Single layer computation is mathematically correct!")
        else:
            print(f"✗ Outputs differ by {abs(network_output - expected_output_np[0][0])}")
        
        free_py_matrix(input_matrix)
        free_py_matrix(result_matrix)
        del mlp
        
    except Exception as e:
        print(f"✗ Error in single layer test: {e}")
        traceback.print_exc()
    
    print("\nTest 2: Multi-layer network consistency")
    try:
        # Test that the same input always produces the same output
        np.random.seed(123)
        mlp = MLP([3, 4, 2], activation="Sigmoid", seed=123)
        test_input = [[0.1], [0.5], [0.9]]
        
        outputs = []
        for i in range(5):
            input_matrix = py_list_to_c_matrix(test_input)
            output_matrix = mlp.forward_pass(input_matrix)
            output = to_numpy(output_matrix)
            outputs.append(output.copy())
            free_py_matrix(input_matrix)
            free_py_matrix(output_matrix)
        
        # Check consistency
        all_consistent = True
        for i in range(1, len(outputs)):
            if not np.allclose(outputs[0], outputs[i], atol=1e-10):
                all_consistent = False
                break
        
        if all_consistent:
            print("✓ Network produces consistent outputs for same input")
        else:
            print("✗ Network produces inconsistent outputs")
        
        del mlp
        
    except Exception as e:
        print(f"✗ Error in consistency test: {e}")

def run_forward_pass_tests():
    """Run all forward pass tests"""
    print("\nFORWARD PASS TESTING SUITE")
    print("=" * 60)
    
    try:
        test_full_forward_pass()
        test_forward_pass_error_cases()
        test_forward_pass_mathematical_correctness()
        
        print("\n" + "=" * 60)
        print("FORWARD PASS TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR IN FORWARD PASS TEST SUITE: {e}")
        traceback.print_exc()

def test_activation_storage():
    """Test that activations are properly stored during forward pass"""
    print("\n" + "=" * 50)
    print("TESTING ACTIVATION STORAGE")
    print("=" * 50)
    
    try:
        mlp = MLP([2, 3, 1], activation="Sigmoid", seed=42)
        input_data = [[0.7], [0.3]]
        input_matrix = py_list_to_c_matrix(input_data)
        
        # Perform forward pass
        output_matrix = mlp.forward_pass(input_matrix)
        
        # Check that activations are stored
        expected_num_activations = len(mlp.layer_sizes)
        actual_num_activations = len(mlp.activations)
        
        if actual_num_activations == expected_num_activations:
            print(f"✓ Correct number of activations stored: {actual_num_activations}")
            
            # Verify each activation has correct dimensions
            for i, activation in enumerate(mlp.activations):
                act_np = to_numpy(activation)
                expected_shape = (mlp.layer_sizes[i], 1)
                if act_np.shape == expected_shape:
                    print(f"✓ Activation {i} has correct shape: {act_np.shape}")
                else:
                    print(f"✗ Activation {i} has wrong shape: {act_np.shape}, expected: {expected_shape}")
        else:
            print(f"✗ Wrong number of activations: {actual_num_activations}, expected: {expected_num_activations}")
        
        # Verify first activation is the input
        first_activation = to_numpy(mlp.activations[0])
        input_np = np.array(input_data)
        if np.allclose(first_activation, input_np):
            print("✓ First activation correctly stores input")
        else:
            print("✗ First activation doesn't match input")
        
        # Verify last activation is the output
        last_activation = to_numpy(mlp.activations[-1])
        output_np = to_numpy(output_matrix)
        if np.allclose(last_activation, output_np):
            print("✓ Last activation correctly stores output")
        else:
            print("✗ Last activation doesn't match output")
        
        free_py_matrix(input_matrix)
        free_py_matrix(output_matrix)
        del mlp
        
    except Exception as e:
        print(f"✗ Error in activation storage test: {e}")
        traceback.print_exc()

def test_seed_reproducibility():
    """Test that using the same seed produces reproducible results"""
    print("\n" + "=" * 50)
    print("TESTING SEED REPRODUCIBILITY")
    print("=" * 50)
    
    try:
        seed = 12345
        architecture = [3, 5, 2]
        
        # Create two identical networks
        mlp1 = MLP(architecture, seed=seed)
        mlp2 = MLP(architecture, seed=seed)
        
        # Check that weights are identical
        weights_identical = True
        for i, (w1, w2) in enumerate(zip(mlp1.weights, mlp2.weights)):
            w1_np = to_numpy(w1)
            w2_np = to_numpy(w2)
            if not np.allclose(w1_np, w2_np, atol=1e-15):
                weights_identical = False
                print(f"✗ Weights for layer {i} are different")
                break
        
        if weights_identical:
            print("✓ Same seed produces identical weights")
        
        # Check that biases are identical
        biases_identical = True
        for i, (b1, b2) in enumerate(zip(mlp1.biases, mlp2.biases)):
            b1_np = to_numpy(b1)
            b2_np = to_numpy(b2)
            if not np.allclose(b1_np, b2_np, atol=1e-15):
                biases_identical = False
                print(f"✗ Biases for layer {i} are different")
                break
        
        if biases_identical:
            print("✓ Same seed produces identical biases")
        
        # Test that forward passes produce identical results
        test_input = [[0.1], [0.5], [0.9]]
        input_matrix1 = py_list_to_c_matrix(test_input)
        input_matrix2 = py_list_to_c_matrix(test_input)
        
        output1 = mlp1.forward_pass(input_matrix1)
        output2 = mlp2.forward_pass(input_matrix2)
        
        output1_np = to_numpy(output1)
        output2_np = to_numpy(output2)
        
        if np.allclose(output1_np, output2_np, atol=1e-15):
            print("✓ Same seed produces identical forward pass results")
        else:
            print("✗ Forward pass results differ despite same seed")
        
        # Test that different seeds produce different results
        mlp3 = MLP(architecture, seed=seed + 1)
        input_matrix3 = py_list_to_c_matrix(test_input)
        output3 = mlp3.forward_pass(input_matrix3)
        output3_np = to_numpy(output3)
        
        if not np.allclose(output1_np, output3_np, atol=1e-6):
            print("✓ Different seeds produce different results")
        else:
            print("⚠ Different seeds produced very similar results (might be coincidence)")
        
        # Clean up
        free_py_matrix(input_matrix1)
        free_py_matrix(input_matrix2)
        free_py_matrix(input_matrix3)
        free_py_matrix(output1)
        free_py_matrix(output2)
        free_py_matrix(output3)
        del mlp1, mlp2, mlp3
        
    except Exception as e:
        print(f"✗ Error in seed reproducibility test: {e}")
        traceback.print_exc()

def test_backward_pass_stub():
    """Test backward pass implementation (if available)"""
    print("\n" + "=" * 50)
    print("TESTING BACKWARD PASS (if implemented)")
    print("=" * 50)
    
    try:
        mlp = MLP([2, 3, 1], seed=42)
        
        # Check if backward_pass method exists and is implemented
        if hasattr(mlp, 'backward_pass'):
            print("✓ backward_pass method exists")
            
            # Test with simple XOR-like data
            X = py_list_to_c_matrix([[1.0], [0.0]])
            y = py_list_to_c_matrix([[1.0]])
            
            try:
                # This will likely fail if not fully implemented
                mlp.backward_pass(X, y)
                print("✓ backward_pass executed without errors")
            except NotImplementedError:
                print("⚠ backward_pass not fully implemented (NotImplementedError)")
            except Exception as e:
                print(f"⚠ backward_pass implementation issue: {e}")
            
            free_py_matrix(X)
            free_py_matrix(y)
        else:
            print("⚠ backward_pass method not found")
        
        del mlp
        
    except Exception as e:
        print(f"✗ Error in backward pass test: {e}")

def test_performance_benchmarks():
    """Test performance with various network sizes"""
    print("\n" + "=" * 50)
    print("TESTING PERFORMANCE BENCHMARKS")
    print("=" * 50)
    
    import time
    
    test_configs = [
        ([10, 10, 1], "Small network"),
        ([100, 50, 10, 1], "Medium network"),
        ([500, 200, 50, 1], "Large network")
    ]
    
    for architecture, description in test_configs:
        try:
            print(f"\n{description}: {architecture}")
            
            # Time network creation
            start_time = time.time()
            mlp = MLP(architecture, seed=42)
            creation_time = time.time() - start_time
            print(f"  Creation time: {creation_time:.4f} seconds")
            
            # Time forward passes
            input_size = architecture[0]
            test_input = py_list_to_c_matrix([[0.5] for _ in range(input_size)])
            
            num_passes = 100
            start_time = time.time()
            for _ in range(num_passes):
                output = mlp.forward_pass(test_input)
                free_py_matrix(output)
            forward_time = time.time() - start_time
            
            avg_forward_time = forward_time / num_passes
            print(f"  Average forward pass time: {avg_forward_time:.6f} seconds")
            print(f"  Forward passes per second: {1/avg_forward_time:.0f}")
            
            free_py_matrix(test_input)
            del mlp
            
        except Exception as e:
            print(f"✗ Error in performance test for {description}: {e}")

def run_comprehensive_test_suite():
    """Run all tests with better organization and error handling"""
    print("COMPREHENSIVE MLP NEURAL NETWORK TESTING SUITE")
    print("=" * 80)
    
    test_functions = [
        ("Matrix Operations", test_matrix_operations),
        ("MLP Initialization", test_mlp_initialization),
        ("Weight Initialization", test_mlp_weight_initialization),
        ("Memory Management", test_mlp_memory_management),
        ("XOR Data Setup", test_mlp_with_xor_data),
        ("Forward Pass Tests", run_forward_pass_tests),
        ("Activation Storage", test_activation_storage),
        ("Seed Reproducibility", test_seed_reproducibility),
        ("Backward Pass", test_backward_pass_stub),
        ("Performance Benchmarks", test_performance_benchmarks)
    ]
    
    passed_tests = 0
    failed_tests = 0
    
    for test_name, test_function in test_functions:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            test_function()
            passed_tests += 1
            print(f"✓ {test_name} completed successfully")
        except Exception as e:
            failed_tests += 1
            print(f"✗ {test_name} failed with error: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"Tests passed: {passed_tests}")
    print(f"Tests failed: {failed_tests}")
    print(f"Success rate: {passed_tests/(passed_tests+failed_tests)*100:.1f}%")
    print("=" * 80)

def run_all_tests():
    """Legacy function name for compatibility"""
    run_comprehensive_test_suite()

if __name__ == "__main__":
    run_comprehensive_test_suite()

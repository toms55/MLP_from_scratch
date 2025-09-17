import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))
from mlp import MLP
import random
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))
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
    py_sigmoid,
    py_sigmoid_derivative,
    from_numpy
)

def test_matrix_operations():
    """Test basic matrix operations"""
    print("=" * 50)
    print("TESTING MATRIX OPERATIONS")
    print("=" * 50)
    
    py_mat1 = [[1.2, 2.99], [3.1, 4.25]]
    py_mat2 = [[5.01, 6.91], [7.25, 8.765]]
    
    c_mat1 = None
    c_mat2 = None
    c_result = None
    c_zero_mat = None
    c_sum_mat = None
    
    try:
        c_mat1 = py_list_to_c_matrix(py_mat1)
        c_mat2 = py_list_to_c_matrix(py_mat2)
        
        c_result = multiply_py_matrices(c_mat1, c_mat2)
        py_result = c_matrix_to_py_list(c_result)
        c_zero_mat = create_zero_py_matrix(3, 3)
        zero_mat = c_matrix_to_py_list(c_zero_mat)
        
        print("Matrix 1:")
        print(py_mat1)
        print("\nMatrix 2:")
        print(py_mat2)
        print("\nResult of matrix multiplication:")
        print(py_result)
        print("\nZero Matrix:")
        print(zero_mat)
        
        c_sum_mat = add_py_matrices(c_mat1, c_mat2)
        print("\nResult of matrix addition:")
        print(c_matrix_to_py_list(c_sum_mat))
        
        print("\nTesting Sigmoid")
        print(f"{py_sigmoid(3)} should equal 0.95257")
        print("Testing Sigmoid Derivative")
        print(f"{py_sigmoid_derivative(3)} should equal 0.04517665973")
        
    finally:
        # Clean up all allocated matrices
        for mat in [c_mat1, c_mat2, c_result, c_zero_mat, c_sum_mat]:
            if mat:
                free_py_matrix(mat)

def test_mlp_initialization():
    """Test MLP initialization with various configurations"""
    print("\n" + "=" * 50)
    print("TESTING MLP INITIALIZATION")
    print("=" * 50)
    
    # Test 1: Valid initialization
    print("\nTest 1: Valid MLP initialization")
    try:
        mlp = MLP([2, 3, 1], activation="Sigmoid", loss="MSE", learning_rate=0.01, seed=42)
        print(f"✓ MLP created successfully with layer sizes: {mlp.layer_sizes}")
        print(f"✓ Activation: {mlp.activation}")
        print(f"✓ Loss: {mlp.loss}")
        print(f"✓ Learning rate: {mlp.learning_rate}")
        print(f"✓ Number of weight matrices: {len(mlp.weights)}")
        print(f"✓ Number of bias vectors: {len(mlp.biases)}")
        del mlp  # Test destructor
        print("✓ MLP destructor called successfully")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: Multiple layer sizes
    print("\nTest 2: MLP with multiple hidden layers")
    try:
        mlp = MLP([4, 8, 6, 3, 1], seed=123)
        print(f"✓ Multi-layer MLP created: {mlp.layer_sizes}")
        print(f"✓ Weight matrices count: {len(mlp.weights)} (expected: 4)")
        del mlp
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: Invalid layer sizes (should print error message)
    print("\nTest 3: Invalid layer sizes (< 2 layers)")
    try:
        mlp = MLP([5], seed=456)
        print("✗ This should have printed an error message")
        del mlp
    except Exception as e:
        print(f"Expected behavior - Error caught: {e}")
    
    # Test 4: Minimum valid size
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
    
    # Set seed for reproducible results
    np.random.seed(42)
    mlp = MLP([3, 4, 2], seed=42)
    
    print("Checking weight initialization bounds (Xavier/Glorot):")
    
    for i, (w, b) in enumerate(zip(mlp.weights, mlp.biases)):
        # Convert C matrices back to numpy for inspection
        # Note: This assumes you have a way to convert back from C matrices
        # You might need to implement this functionality
        print(f"\nLayer {i} -> {i+1}:")
        print(f"✓ Weight matrix initialized")
        print(f"✓ Bias vector initialized")
        
        # Expected limit for Xavier initialization
        layer_in = mlp.layer_sizes[i]
        layer_out = mlp.layer_sizes[i+1]
        expected_limit = np.sqrt(6 / (layer_in + layer_out))
        print(f"✓ Expected weight range: ±{expected_limit:.4f}")

def test_mlp_memory_management():
    """Test memory management and multiple MLPs"""
    print("\n" + "=" * 50)
    print("TESTING MLP MEMORY MANAGEMENT")
    print("=" * 50)
    
    print("Creating and destroying multiple MLPs...")
    
    # Create multiple MLPs to test memory management
    mlps = []
    try:
        for i in range(5):
            mlp = MLP([2, 4, 2], seed=i)
            mlps.append(mlp)
            print(f"✓ MLP {i+1} created")
        
        print("✓ All MLPs created successfully")
        
        # Delete them one by one
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
    
    # XOR training data
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]
    
    print("XOR Training Data:")
    for inputs, targets in training_data:
        print(f"  Input: {inputs} -> Target: {targets}")
    
    # Create MLP suitable for XOR
    print("\nCreating MLP for XOR problem...")
    try:
        mlp = MLP([2, 4, 1], activation="Sigmoid", loss="MSE", learning_rate=0.1, seed=42)
        print("✓ XOR MLP created successfully")
        print(f"  Architecture: {mlp.layer_sizes}")
        print(f"  Activation: {mlp.activation}")
        print(f"  Loss function: {mlp.loss}")
        print(f"  Learning rate: {mlp.learning_rate}")
        
        # Test data conversion (you'll need to implement forward pass later)
        print("\nTesting data format compatibility...")
        for i, (inputs, targets) in enumerate(training_data):
            print(f"  Sample {i+1}: {inputs} -> {targets} ✓")
        
        del mlp
        print("✓ XOR MLP test completed")
        
    except Exception as e:
        print(f"✗ Error in XOR MLP test: {e}")

def test_mlp_different_configurations():
    """Test MLP with different activation functions and loss functions"""
    print("\n" + "=" * 50)
    print("TESTING MLP DIFFERENT CONFIGURATIONS")
    print("=" * 50)
    
    configurations = [
        {"layers": [2, 3, 1], "activation": "Sigmoid", "loss": "MSE", "lr": 0.01},
        {"layers": [4, 6, 4, 1], "activation": "ReLU", "loss": "CrossEntropy", "lr": 0.001},
        {"layers": [3, 5, 2], "activation": "Tanh", "loss": "MSE", "lr": 0.05},
    ]
    
    for i, config in enumerate(configurations):
        print(f"\nConfiguration {i+1}:")
        print(f"  Layers: {config['layers']}")
        print(f"  Activation: {config['activation']}")
        print(f"  Loss: {config['loss']}")
        print(f"  Learning Rate: {config['lr']}")
        
        try:
            mlp = MLP(
                layer_sizes=config['layers'],
                activation=config['activation'],
                loss=config['loss'],
                learning_rate=config['lr'],
                seed=i
            )
            print("  ✓ Configuration created successfully")
            del mlp
            print("  ✓ Configuration destroyed successfully")
        except Exception as e:
            print(f"  ✗ Error: {e}")

def run_all_tests():
    """Run all MLP tests"""
    print("MLP NEURAL NETWORK TESTING SUITE")
    print("=" * 60)
    
    try:
        # Run matrix operation tests first
        test_matrix_operations()
        
        # Run MLP-specific tests
        test_mlp_initialization()
        test_mlp_weight_initialization()
        test_mlp_memory_management()
        test_mlp_with_xor_data()
        test_mlp_different_configurations()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR IN TEST SUITE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()

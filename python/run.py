import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPRegressor
from mlp import MLP
import c_wrapper
from typing import Tuple, Optional, Dict, List

def custom_train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.1, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if random_state is not None:
        np.random.seed(random_state)
    
    data = np.hstack((X, y))
    np.random.shuffle(data)
    
    n_samples = data.shape[0]
    n_test = int(n_samples * test_size)
    
    test_data = data[:n_test]
    train_data = data[n_test:]
    
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1].reshape(-1, 1)
    
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1].reshape(-1, 1)
    
    return X_train, X_test, y_train, y_test

def custom_standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1e-8
    
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    return X_train_scaled, X_test_scaled

def benchmark_custom_mlp(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int = 500) -> Dict:
    """Benchmark custom MLP implementation"""
    print("\n" + "="*60)
    print("BENCHMARKING CUSTOM MLP")
    print("="*60)
    
    input_size = X_train.shape[1]
    mlp = MLP(
        layer_sizes=[input_size, 10, 10, 5, 1],
        hidden_activation="ReLU",
        output_activation="relu",
        loss="MSE",
        learning_rate=0.001,
        seed=32
    )
    
    # Training time
    start_train = time.time()
    mlp.train_model(X_train, y_train, epochs=epochs)
    train_time = time.time() - start_train
    
    # Inference time
    start_infer = time.time()
    predictions = mlp.predict(X_test)
    infer_time = time.time() - start_infer
    
    # Calculate MSE
    y_test_c = c_wrapper.transpose_py_matrix(c_wrapper.from_numpy(y_test))
    predictions_c = c_wrapper.transpose_py_matrix(c_wrapper.from_numpy(predictions))
    mse = c_wrapper.py_mean_squared_error(y_test_c, predictions_c)
    c_wrapper.free_py_matrix(y_test_c)
    c_wrapper.free_py_matrix(predictions_c)
    
    print(f"Training Time: {train_time:.4f} seconds")
    print(f"Inference Time: {infer_time:.4f} seconds")
    print(f"MSE: {mse:.4f}")
    
    return {
        'name': 'Custom MLP',
        'train_time': train_time,
        'infer_time': infer_time,
        'mse': mse,
        'total_time': train_time + infer_time
    }

def benchmark_sklearn_mlp(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int = 500) -> Dict:
    """Benchmark scikit-learn MLPRegressor"""
    print("\n" + "="*60)
    print("BENCHMARKING SCIKIT-LEARN MLP (SGD)")
    print("="*60)
    
    mlp = MLPRegressor(
        hidden_layer_sizes=(10, 10, 5),
        activation='relu',
        solver='sgd',  # Use simple gradient descent
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=epochs,
        random_state=32,
        verbose=False
    )
    
    # Training time
    start_train = time.time()
    mlp.fit(X_train, y_train.ravel())
    train_time = time.time() - start_train
    
    # Inference time
    start_infer = time.time()
    predictions = mlp.predict(X_test)
    infer_time = time.time() - start_infer
    
    # Calculate MSE
    mse = np.mean((y_test.ravel() - predictions) ** 2)
    
    print(f"Training Time: {train_time:.4f} seconds")
    print(f"Inference Time: {infer_time:.4f} seconds")
    print(f"MSE: {mse:.4f}")
    
    return {
        'name': 'Scikit-Learn MLP',
        'train_time': train_time,
        'infer_time': infer_time,
        'mse': mse,
        'total_time': train_time + infer_time
    }

def benchmark_pytorch_mlp(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int = 500) -> Dict:
    """Benchmark PyTorch neural network"""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        print("\n" + "="*60)
        print("BENCHMARKING PYTORCH MLP (SGD)")
        print("="*60)
        
        # Define PyTorch model
        class PyTorchMLP(nn.Module):
            def __init__(self, input_size):
                super(PyTorchMLP, self).__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, 10),
                    nn.ReLU(),
                    nn.Linear(10, 10),
                    nn.ReLU(),
                    nn.Linear(10, 5),
                    nn.ReLU(),
                    nn.Linear(5, 1)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        # Set seed for reproducibility
        torch.manual_seed(32)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        
        model = PyTorchMLP(X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)  # Simple gradient descent
        
        # Training time
        start_train = time.time()
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        train_time = time.time() - start_train
        
        # Inference time
        model.eval()
        start_infer = time.time()
        with torch.no_grad():
            predictions = model(X_test_tensor).numpy()
        infer_time = time.time() - start_infer
        
        # Calculate MSE
        mse = np.mean((y_test - predictions) ** 2)
        
        print(f"Training Time: {train_time:.4f} seconds")
        print(f"Inference Time: {infer_time:.4f} seconds")
        print(f"MSE: {mse:.4f}")
        
        return {
            'name': 'PyTorch MLP (SGD)',
            'train_time': train_time,
            'infer_time': infer_time,
            'mse': mse,
            'total_time': train_time + infer_time
        }
    
    except ImportError:
        print("\nPyTorch not installed. Skipping PyTorch benchmark.")
        return None

def plot_benchmark_results(results: List[Dict]):
    """Create visualization of benchmark results"""
    results = [r for r in results if r is not None]
    
    if not results:
        print("No results to plot")
        return
    
    names = [r['name'] for r in results]
    train_times = [r['train_time'] for r in results]
    infer_times = [r['infer_time'] for r in results]
    mses = [r['mse'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training time comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(names, train_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_ylabel('Time (seconds)', fontsize=11)
    ax1.set_title('Training Time Comparison', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{train_times[i]:.2f}s',
                ha='center', va='bottom', fontsize=10)
    
    # Inference time comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(names, infer_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_ylabel('Time (seconds)', fontsize=11)
    ax2.set_title('Inference Time Comparison', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{infer_times[i]:.4f}s',
                ha='center', va='bottom', fontsize=10)
    
    # MSE comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(names, mses, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax3.set_ylabel('Mean Squared Error', fontsize=11)
    ax3.set_title('Model Performance (MSE)', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='x', rotation=15)
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{mses[i]:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    # Speed comparison (relative to custom MLP)
    ax4 = axes[1, 1]
    custom_time = train_times[0]
    speedups = [custom_time / t for t in train_times]
    bars4 = ax4.bar(names, speedups, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax4.set_ylabel('Speedup Factor', fontsize=11)
    ax4.set_title('Training Speed Relative to Custom MLP', fontsize=13, fontweight='bold')
    ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Custom MLP baseline')
    ax4.tick_params(axis='x', rotation=15)
    ax4.legend()
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedups[i]:.2f}x',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('mlp_benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_table(results: List[Dict]):
    """Print a summary table of all results"""
    results = [r for r in results if r is not None]
    
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY TABLE")
    print("="*80)
    print(f"{'Model':<25} {'Train (s)':<15} {'Infer (s)':<15} {'MSE':<15} {'Total (s)':<15}")
    print("-"*80)
    
    for r in results:
        print(f"{r['name']:<25} {r['train_time']:<15.4f} {r['infer_time']:<15.4f} {r['mse']:<15.4f} {r['total_time']:<15.4f}")
    
    print("-"*80)
    
    # Calculate speedup factors
    if len(results) > 1:
        custom_train = results[0]['train_time']
        print("\nSpeedup Factors (relative to Custom MLP):")
        print("-"*80)
        for r in results:
            speedup = custom_train / r['train_time']
            print(f"{r['name']:<25} {speedup:.2f}x")
    
    print("="*80)

def main():
    """Main benchmark execution"""
    print("="*80)
    print("MLP IMPLEMENTATION BENCHMARK")
    print("="*80)
    print("\nLoading California Housing Dataset...")
    
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = custom_train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    
    print("Scaling features...")
    X_train_scaled, X_test_scaled = custom_standard_scaler(X_train, X_test)
    
    print(f"\nDataset Info:")
    print(f"Training samples: {X_train_scaled.shape[0]}")
    print(f"Test samples: {X_test_scaled.shape[0]}")
    print(f"Features: {X_train_scaled.shape[1]}")
    print(f"Epochs: 500")
    
    # Run benchmarks
    epochs = 500
    results = []
    
    results.append(benchmark_custom_mlp(X_train_scaled, y_train, X_test_scaled, y_test, epochs))
    results.append(benchmark_sklearn_mlp(X_train_scaled, y_train, X_test_scaled, y_test, epochs))
    results.append(benchmark_pytorch_mlp(X_train_scaled, y_train, X_test_scaled, y_test, epochs))
    
    # Display results
    print_summary_table(results)
    plot_benchmark_results(results)

if __name__ == "__main__":
    main()

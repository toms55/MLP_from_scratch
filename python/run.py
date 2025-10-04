import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from mlp import MLP
import c_wrapper
from typing import Tuple, Optional

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

print("Loading California Housing Dataset...")
housing = fetch_california_housing()
X = housing.data
y = housing.target.reshape(-1, 1)

X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.1, random_state=42)

print("Scaling features manually...")
X_train_scaled, X_test_scaled = custom_standard_scaler(X_train, X_test)

input_size = X_train_scaled.shape[1]
mlp = MLP(layer_sizes=[input_size, 20, 10, 1], hidden_activation="ReLU", output_activation="relu", loss="MSE", learning_rate=0.001, seed=32)

print("Starting training...")
mlp.train_model(X_train_scaled, y_train, epochs=300)


print("\n--- Testing the trained model ---")
predictions = mlp.predict(X_test_scaled)

y_test_c = c_wrapper.transpose_py_matrix(c_wrapper.from_numpy(y_test))
predictions_c = c_wrapper.transpose_py_matrix(c_wrapper.from_numpy(predictions))

mse = c_wrapper.py_mean_squared_error(y_test_c, predictions_c)
rmse = np.sqrt(mse)

y_true = y_test.flatten()
predictions_flat = predictions.flatten()
non_zero_indices = y_true != 0

abs_percentage_errors = np.abs(
    (y_true[non_zero_indices] - predictions_flat[non_zero_indices]) / y_true[non_zero_indices]
)
mape = np.mean(abs_percentage_errors) * 100 # move to C implementation

print("mape", mape)

c_wrapper.free_py_matrix(y_test_c)
c_wrapper.free_py_matrix(predictions_c)

print(f"\nModel Evaluation (Regression Task):")
print(f"Mean Squared Error (MSE) on Test Set: {mse:.4f}")
print(f"Root Mean Squared Error (RMSE) on Test Set: {rmse:.4f}")

print("\nSample Predictions (Values in $100,000s):")
for i in range(10):
    true_value = y_test[i][0]
    pred_value = predictions[i][0]
    print(f"Sample {i+1}: True Value: {true_value:.2f}, Predicted Value: {pred_value:.2f}")



plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("True Values ($100k)")
plt.ylabel("Predictions ($100k)")
plt.title("True vs. Predicted Median House Value")
plt.grid(True)
plt.show()

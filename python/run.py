import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlp import MLP 
import c_wrapper

print("Loading California Housing Dataset...")
housing = fetch_california_housing()
X = housing.data
y = housing.target.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Scaling features...")
scaler = StandardScaler()
mlp = MLP(layer_sizes=[8, 16, 16, 5, 1], hidden_activation="ReLU", output_activation="identity", learning_rate=0.001)

print("Starting training...")
mlp.train_model(X_train, y_train, epochs=150)


print("\n--- Testing the trained model ---")
predictions = mlp.predict(X_test)

y_test_c = c_wrapper.transpose_py_matrix(c_wrapper.from_numpy(y_test))
predictions_c = c_wrapper.transpose_py_matrix(c_wrapper.from_numpy(predictions))

mse = c_wrapper.py_mean_squared_error(y_test_c, predictions_c)
rmse = np.sqrt(mse)

c_wrapper.free_py_matrix(y_test_c)
c_wrapper.free_py_matrix(predictions_c)

print(f"\nModel Evaluation (Regression Task):")
print(f"Mean Squared Error (MSE) on Test Set: {mse:.4f}")
print(f"Root Mean Squared Error (RMSE) on Test Set: {rmse:.4f}")

print("\nSample Predictions:")
for i in range(10):
    true_price = y_test[i][0] * 100000 # Convert back to standard dollars (approx)
    pred_price = predictions[i][0] * 100000 # Convert back to standard dollars (approx)
    
    print(f"True House Value: ${true_price:,.2f} | Predicted Value: ${pred_price:,.2f}")

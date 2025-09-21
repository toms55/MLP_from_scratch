import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from mlp import MLP

n_samples = 1000
centers = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
X, y_blobs = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.4, random_state=42)

y = np.array([0 if label in [0, 3] else 1 for label in y_blobs])
y = y.reshape(-1, 1) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train[:,0] == 0][:, 0], X_train[y_train[:,0] == 0][:, 1], label='Class 0')
plt.scatter(X_train[y_train[:,0] == 1][:, 0], X_train[y_train[:,0] == 1][:, 1], label='Class 1')
plt.title("Generated XOR-like Training Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

mlp = MLP(layer_sizes=[2, 4, 5, 1], learning_rate=0.1)

print("Starting training...")
mlp.train_model(X_train, y_train, epochs=35)

print("\n--- Testing the trained model ---")
predictions = mlp.predict(X_test)

correct_predictions = 0
for i in range(len(X_test)):
    raw_prediction = predictions[i][0]
    classified_prediction = 1 if raw_prediction > 0.5 else 0
    true_label = y_test[i][0]
    
    if classified_prediction == true_label:
        correct_predictions += 1
    
    if i < 5:
        print(f"Input: {X_test[i]}, True: {true_label}, Pred: {raw_prediction:.4f} -> Classified as {classified_prediction}")

accuracy = (correct_predictions / len(y_test)) * 100
print(f"\nModel Accuracy on Test Set: {accuracy:.2f}%")

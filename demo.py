import numpy as np
import matplotlib.pyplot as plt
def true_fun(X):
    return np.cos(1.5 * np.pi * X)

# Generating the dataset
np.random.seed(40)
n_samples = 15
X_train = np.sort(np.random.rand(n_samples))
y_train = true_fun(X_train) + np.random.rand(n_samples) *0.1

X_test = np.linspace(0,1,100)
y_test = true_fun(X_test)
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(X_test, y_test, label="True Risk Reference (The 'Ground Truth')", color='green', linestyle='--')
plt.scatter(X_train, y_train, edgecolor='b', s=40, label="Training Dataset (The 'Noisy Reality')")
plt.xlabel("Input (X)")
plt.ylabel("Output (y)")
plt.title("The Gap Between Your Dataset and The Truth")
plt.legend(loc="best")
plt.show()


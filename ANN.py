import numpy as np
#single perceptron by Bishnu Sharma
class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None


    def activation(self, z):
        return 1 if z >= 0 \
            else 0  # Step function

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation(linear_output)
                error = y[idx] - y_pred
                self.weights += self.lr * error * x_i
                self.bias += self.lr * error

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return [self.activation(z) for z in linear_output]

# Define the logic gates
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])  # AND
y_or = np.array([0, 1, 1, 1])   # OR
y_xor = np.array([0, 1, 1, 0])   # XOR (not linearly separable)

# Train and test AND gate
print("\n--- AND Gate ---")
perceptron_and = Perceptron()
perceptron_and.fit(X, y_and)
print("Weights:", perceptron_and.weights, "Bias:", perceptron_and.bias)
print("Predictions:", perceptron_and.predict(X))

# Train and test OR gate
print("\n--- OR Gate ---")
perceptron_or = Perceptron()
perceptron_or.fit(X, y_or)
print("Weights:", perceptron_or.weights, "Bias:", perceptron_or.bias)
print("Predictions:", perceptron_or.predict(X))

# Try XOR (will fail)
print("\n--- XOR Gate (Will Fail) ---")
perceptron_xor = Perceptron(epochs=1000)  # Even with more epochs, it fails
perceptron_xor.fit(X, y_xor)
print("Weights:", perceptron_xor.weights, "Bias:", perceptron_xor.bias)
print("Predictions:", perceptron_xor.predict(X))
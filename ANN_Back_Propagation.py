import numpy as np
#backpropagation by Bishnu Shama
class NeuralNetwork:
    def __init__(self):
        # Network architecture (2-2-1)
        self.input_size = 2
        self.hidden_size = 2
        self.output_size = 1

        # Initialize weights with random values
        self.W1 = np.random.randn(self.input_size, self.hidden_size)  # Input to hidden
        self.W2 = np.random.randn(self.hidden_size, self.output_size)  # Hidden to output
        self.b1 = np.zeros((1, self.hidden_size))  # Hidden layer bias
        self.b2 = np.zeros((1, self.output_size))  # Output layer bias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)  # Hidden layer activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)  # Output layer activation
        return self.a2

    def backward(self, X, y, output, learning_rate=0.1):
        # Backward propagation
        self.error = y - output
        self.delta2 = self.error * self.sigmoid_derivative(output)
        self.error_hidden = self.delta2.dot(self.W2.T)
        self.delta1 = self.error_hidden * self.sigmoid_derivative(self.a1)

        # Update weights and biases
        self.W2 += self.a1.T.dot(self.delta2) * learning_rate
        self.b2 += np.sum(self.delta2, axis=0, keepdims=True) * learning_rate
        self.W1 += X.T.dot(self.delta1) * learning_rate
        self.b1 += np.sum(self.delta1, axis=0) * learning_rate

    def train(self, X, y, epochs=10000, learning_rate=0.1):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)

            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return np.round(self.forward(X))


# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train the neural network
nn = NeuralNetwork()
print("Initial predictions (before training):")
print(nn.predict(X))

nn.train(X, y, epochs=10000, learning_rate=0.1)

print("\nFinal predictions (after training):")
print(nn.predict(X))
print("\nFinal weights and biases:")
print("W1:", nn.W1)
print("b1:", nn.b1)
print("W2:", nn.W2)
print("b2:", nn.b2)
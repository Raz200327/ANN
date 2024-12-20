import numpy as np
import pickle

class NeuralNetwork:
    def loadModel(self, path=""):
        self.pre_activations = []
        self.post_activation = []
        with open(path, "rb") as file:
            params = pickle.load(file)
        self.weights = params["weights"]
        self.biases = params["biases"]
        self.lr = params["lr"]
            
    def createNetwork(self, input, hidden_layers, output_size, num_neurons, lr=0.001):
        self.pre_activations = []
        self.post_activation = []
        self.weights = []
        self.biases = []
        self.weights.append(np.random.randn(input, num_neurons)/2)
        self.biases.append(np.random.uniform(-1, 1, (1, num_neurons))/2)
        for _ in range(hidden_layers-2):
            self.weights.append(np.random.randn(num_neurons, num_neurons)/2)
            self.biases.append(np.random.uniform(-1, 1, (1, num_neurons))/2)
        self.weights.append(np.random.randn(num_neurons, output_size)/2)
        self.biases.append(np.random.uniform(-1, 1, (1, output_size))/2)
        self.lr = lr
    def ReLU(self, x):
        return np.maximum(0, x)
    
    def ReLUGrad(self, x):
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def crossEntropy(self, pred, true):
        return -np.sum(true * np.log(pred + 1e-15)) / pred.shape[0]
    
    def forward(self, x):
        self.input = x
        for layer in range(len(self.weights)-1):
            x = np.dot(x, self.weights[layer]) + self.biases[layer]
            self.pre_activations.append(x)
            relu = self.ReLU(x)
            self.post_activation.append(relu)
            x = relu
        x = np.dot(x, self.weights[layer+1] + self.biases[layer+1])
        self.pre_activations.append(x)
        self.output = self.softmax(x)

        return self.output
    
    def backwards(self, y_hat, y):
        batch_size = y.shape[0]
        # Correct the gradient of the output layer
        logits = y_hat - y

        d_final_W = np.dot(self.post_activation[-1].T, logits) / batch_size
        d_final_b = np.sum(logits, axis=0) / batch_size
        self.weights[-1] -= self.lr * d_final_W
        self.biases[-1] -= self.lr * d_final_b

        # Backpropagate through the hidden layers
        delta = np.dot(logits, self.weights[-1].T) * self.ReLUGrad(self.post_activation[-1])
        weight_idx = -1
        for _ in range(len(self.weights)-2):
            weight_idx -= 1
            d_weight = np.dot(self.post_activation[weight_idx].T, delta) / batch_size
            d_bias = np.sum(delta, axis=0) / batch_size
            self.weights[weight_idx] -= self.lr * d_weight
            self.biases[weight_idx] -= self.lr * d_bias
            delta = np.dot(delta, self.weights[weight_idx].T) * self.ReLUGrad(self.post_activation[weight_idx])
        weight_idx -= 1
        d_first_W = np.dot(self.input.T, delta) / batch_size
        d_first_b = np.sum(delta, axis=0) / batch_size
        self.weights[weight_idx] -= self.lr * d_first_W
        self.biases[weight_idx] -= self.lr * d_first_b
        self.post_activation = []
        self.pre_activations = []

    def accuracy(self, y_hat, y):
        predictions = np.argmax(y_hat, axis=1)
        labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == labels)
        return accuracy


    def train(self, X, Y, epochs):
        for epoch in range(epochs):
            results = self.forward(X)
            loss = self.crossEntropy(results, Y)
            accuracy = self.accuracy(results, Y)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss}, Accuracy: {accuracy}")
            self.backwards(results, Y)
    
    def saveWeights(self, name):
        paramDict = {
            "weights": self.weights,
            "biases": self.biases,
            "lr": self.lr
        }
        with open(f"./{name}.pkl", "wb") as file:
            pickle.dump(paramDict, file)
        print(f"Saved parameters to: ./{name}.pkl")




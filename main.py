import numpy as np

class HiddenLayer:
    def __init__(self, weight, input, bias):
        self.weights = weight  # matrix
        self.bias = bias
        self.input = input

class NeuralNetwork:
    def __init__(self):
        self.input = None
        self.targets = None
        self.ih_weights = np.random.rand(10, 784) - 0.5  # creates an array with random values between 0.5 and -0.5
        self.ih_bias = np.random.rand(10, 1) - 0.5
        self.ho_weights = np.random.rand(10, 10) - 0.5
        self.ho_bias = np.random.rand(10, 1) - 0.5

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def calc_hidden(self, weights, bias, input):
        return np.dot(weights, input) + bias

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        A = np.exp(x) / sum(np.exp(x))
        return A

    def deriv_relu(self, x):
        return x > 0

    def get_predictions(self, x):
        return np.argmax(x, 0)

    def get_accuracy(self, x, y):
        return np.sum(x == y) / y.size

    def train(self, train, label, m):
        """feed forward"""
        # input - hidden
        self.input = train
        self.targets = label
        input_hidden = HiddenLayer(self.ih_weights, self.input, self.ih_bias)
        input_calc = self.calc_hidden(input_hidden.weights, input_hidden.bias, input_hidden.input)  # Z[1]
        input_activation = self.relu(input_calc)  # A[1]
        # hidden - output
        first_hidden = HiddenLayer(self.ho_weights, input_activation, self.ho_bias)
        first_calc = self.calc_hidden(first_hidden.weights, first_hidden.bias, first_hidden.input)  # Z[2]
        output = self.softmax(first_calc)  # A[2]

        """back prop"""
        # figure out errors of each layer (weights only)
        label = self.one_hot(self.targets)
        output_error = output - label  # matrix -> error of the output dz[2]

        ho_error = 1 / m * output_error.dot(np.transpose(input_activation))  # error of the (hidden - output) weights dw[2]
        bias_ho_error = 1 / m * np.sum(output_error)  # db[2]

        first_hidden_error = np.dot(np.transpose(first_hidden.weights), output_error) * self.deriv_relu(input_calc)  # error of the (input - hidden) output dz[1]
        ih_error = 1 / m * first_hidden_error.dot(np.transpose(input_hidden.input))  # dw[1]
        bias_ih_error = 1 / m * np.sum(first_hidden_error)  # db[1]

        """gradient descent"""
        # update params/learn
        learning_rate = 0.10

        weight_ih_delta = input_hidden.weights - learning_rate * ih_error  # lr * err * x (dw 1)
        bias_ih_delta = input_hidden.bias - learning_rate * bias_ih_error  # lr * err (bias_ih_error)

        weight_ho_delta = first_hidden.weights - learning_rate * ho_error  # lr * err * x (dw 2)
        bias_ho_delta = first_hidden.bias - learning_rate * bias_ho_error  # lr * err (bias_ho_error)

        predictions = self.get_predictions(output)
        accuracy = self.get_accuracy(predictions, self.targets)
        # print("Accuracy: " + str(accuracy))

        return weight_ih_delta, bias_ih_delta, weight_ho_delta, bias_ho_delta, accuracy

    def predict(self, ih_weight, ih_bias, ho_weight, ho_bias, train):
        self.input = train
        # input - hidden
        input_hidden = HiddenLayer(ih_weight, train, ih_bias)
        input_calc = self.calc_hidden(input_hidden.weights, input_hidden.bias, input_hidden.input)  # Z[1]
        input_activation = self.relu(input_calc)  # A[1]
        # hidden - output
        first_hidden = HiddenLayer(ho_weight, input_activation, ho_bias)
        first_calc = self.calc_hidden(first_hidden.weights, first_hidden.bias, first_hidden.input)  # Z[2]
        output = self.softmax(first_calc)  # A[2]
        predictions = self.get_predictions(output)

        return input_activation, output, predictions






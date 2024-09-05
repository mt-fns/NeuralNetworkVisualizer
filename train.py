from main import NeuralNetwork
import numpy as np
import pandas as pd

data = pd.read_csv('digit-recognizer/train.csv')
data = np.array(data)
m, n = data.shape  # m = number of images
np.random.shuffle(data)  # shuffle before splitting into dev and training sets

data_dev = data[0:100].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[100:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

# print(X_train.shape, Y_train[0]) # image is in column not row (1st column = pixels of 1st image)
nn = NeuralNetwork()
accuracy = None

for i in range(3000):
    ih_weight, ih_bias, ho_weight, ho_bias, accuracy = nn.train(X_train, Y_train, m)
    nn.ih_weights = ih_weight
    nn.ih_bias = ih_bias
    nn.ho_weights = ho_weight
    nn.ho_bias = ho_bias
    print(ih_weight, ih_bias, ho_weight, ho_bias)

if accuracy > 0.8:
    print("saved")
    # np.savez_compressed('saved_nn.npz', array1=nn.ih_weights, array2=nn.ih_bias, array3=nn.ho_weights, array4=nn.ho_bias)
    #np.savez("weights_biases", nn.ih_weights, nn.ih_bias, nn.ho_weights, nn.ho_bias)

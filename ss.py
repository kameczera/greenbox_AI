import time
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

def model(X, Y, layers_dims, learning_rate = 0.03, activation = 'relu', num_iterations = 3000):#lr was 0.009

    np.random.seed(1)
    costs = []              
    
    parameters = initialize_parameters(layers_dims)

    for i in range(0, num_iterations):
        AL, forward_cache = forward_propagation(X, parameters, activation)
        cost = compute_cost(AL, Y)
       
    return parameters   

def forward_propagation(X, parameters, activation):
   
    forward_cache = {}
    L = len(parameters) // 2                  
    
    forward_cache['A0'] = X

    for l in range(1, L):
        forward_cache['Z' + str(l)] = parameters['W' + str(l)].dot(forward_cache['A' + str(l-1)]) + parameters['b' + str(l)]
        
        if activation == 'tanh':
            forward_cache['A' + str(l)] = tanh(forward_cache['Z' + str(l)])
        else:
            forward_cache['A' + str(l)] = relu(forward_cache['Z' + str(l)])
        print(f"a: {forward_cache['A' + str(l - 1)].shape}, w: {parameters['W' + str(l)].shape}")
            

    forward_cache['Z' + str(L)] = parameters['W' + str(L)].dot(forward_cache['A' + str(L-1)]) + parameters['b' + str(L)]
    
    if forward_cache['Z' + str(L)].shape[0] == 1:
        forward_cache['A' + str(L)] = sigmoid(forward_cache['Z' + str(L)])
    else :
        forward_cache['A' + str(L)] = softmax(forward_cache['Z' + str(L)])
    
    return forward_cache['A' + str(L)], forward_cache

def compute_cost(AL, Y):
    m = Y.shape[1]
    
    if Y.shape[0] == 1:
        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    else:
        cost = -(1./m) * np.sum(Y * np.log(AL))
        
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    
    return cost

def initialize_parameters(layer_dims):
    
    parameters = {}
    L = len(layer_dims)            

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def softmax(z):
    expZ = np.exp(z)
    return expZ/(np.sum(expZ, 0))

def relu(Z):
    A = np.maximum(0,Z)
    return A

def tanh(x):
    return np.tanh(x)

def derivative_relu(Z):
    return np.array(Z > 0, dtype = 'float')

def derivative_tanh(x):
    return (1 - np.power(x, 2))

if __name__ == '__main__':

    # python pycuda.py X_train.csv Y_train.csv 65536 65536 65536 1
    X_train = sys.argv[1]
    Y_train = sys.argv[2]
    X_train = np.loadtxt(X_train, delimiter = ',').astype(np.float32) / 255.0
    Y_train = np.loadtxt(Y_train, delimiter = ',').astype(np.float32).reshape(1, -1)
    Y_train = Y_train.astype(np.float32)
    layer_dim = [X_train.shape[0]]
    for i in range(3, len(sys.argv)):
        layer_dim.append(int(sys.argv[i]))
    # X_test = np.loadtxt('dataset/cat_test_x.csv', delimiter = ',')/255.0
    # Y_test = np.loadtxt('dataset/cat_test_y.csv', delimiter = ',').reshape(1, X_test.shape[1])
    lr = 0.0075
    iters = 1000
    parameters = model(X_train, Y_train, layer_dim, lr, activation = 'relu', num_iterations = iters)
import time
import pycuda.autoinit
from pycuda import driver, compiler, gpuarray, tools
import sys
import pycuda.driver as cuda
import numpy as np
from cuda_functions import forward_propagation_gpu, tanh_gpu, relu_gpu, sigmoid_gpu, cost_gpu, subtract_gpu, compute_dW, compute_db, compute_dZ_gpu, update_weights_gpu

def model(X_train, Y_train, layers_dim, learning_rate, activation = 'relu', num_iteration = 100):
    L = len(layers_dim)
    mem_access_gpu = initialize_parameters(layers_dim, X_train, Y_train, L)
    for i in range(0, num_iteration):
        forward_propagation(X_train, activation, mem_access_gpu, L)
        cost = compute_cost(mem_access_gpu["Y"], mem_access_gpu["A" + str(L - 1)])
        backward_propagation(mem_access_gpu["A" + str(L - 1)], mem_access_gpu["Y"], mem_access_gpu, L, activation)
        update_parameters(mem_access_gpu, learning_rate, L)

        if i % (num_iteration / 10) == 0:
            print("\niter:{} \t cost: {} \t train_acc:{} \t test_acc:".format(i, np.round(cost, 2), accuracy(X_train, Y_train, mem_access_gpu, activation, L)))
            # accuracy(X_test, Y_test, mem_access_gpu, activation, L)
        if i % ((num_iteration / 10) / 10) == 0:
            print("==", end = '')
    

def accuracy(X, Y, mem_access_gpu, activation, L):
    m =  Y.shape[1]
    
    forward_propagation(X, activation, mem_access_gpu, L)
    results = np.empty((mem_access_gpu["A" + str(L - 1)][1], mem_access_gpu["A" + str(L - 1)][2]), dtype=np.float32)
    cuda.memcpy_dtoh(results, mem_access_gpu["A" + str(L - 1)][0])
    if Y.shape[0] == 1:
        results = np.array(results > 0.5, dtype = 'float')
    else:
        Y = np.argmax(Y, 0)
        results = np.argmax(results, axis = 0)

    return np.round(np.sum(Y == results)/m, 2) 

def initialize_parameters(layers_dim, X, Y, L):
    mem_access_gpu = {}
    mem_access_gpu['A0'] = (gpuarray.to_gpu(X), X.shape[0], X.shape[1])
    mem_access_gpu['Y'] = (gpuarray.to_gpu(Y), Y.shape[0], Y.shape[1])

    for l in range(1, L):
        W_init = np.random.randn(layers_dim[l], layers_dim[l - 1]).astype(np.float32)
        W_scaled = W_init / np.sqrt(layers_dim[l - 1]).astype(np.float32)
        w_rows, w_cols = W_scaled.shape
        mem_access_gpu["W" + str(l)] = (gpuarray.to_gpu(W_scaled), w_rows, w_cols)

        b = np.zeros((layers_dim[l], 1), dtype=np.float32)
        b_rows, b_cols = b.shape
        mem_access_gpu["b" + str(l)] = (gpuarray.to_gpu(b), b_rows, b_cols)
    
    return mem_access_gpu

def backward_propagation(AL, Y, mem_access_gpu, L, activation = 'relu'):
    L = L - 1
    m = AL[2]
    mem_access_gpu["dZ" + str(L)] = subtract_gpu(AL[0], Y[0], Y[2])
    mem_access_gpu["db" + str(L)] = compute_db(mem_access_gpu["dZ" + str(L)], m)
    mem_access_gpu["dW" + str(L)] = compute_dW(mem_access_gpu["dZ" + str(L)], mem_access_gpu["A" + str(L - 1)], m)

    for l in reversed(range(1, L)):
        mem_access_gpu["dZ" + str(l)] = compute_dZ_gpu(mem_access_gpu["W" + str(l + 1)], mem_access_gpu["dZ" + str(l + 1)], mem_access_gpu["A" + str(l)])
        mem_access_gpu["db" + str(l)] = compute_db(mem_access_gpu["dZ" + str(l)], m)
        mem_access_gpu["dW" + str(l)] = compute_dW(mem_access_gpu["dZ" + str(l)], mem_access_gpu["A" + str(l - 1)], m)

def update_parameters(mem_access_gpu, learning_rate, L):
    for l in range(1, L):
        # update_weights_gpu(mem_access_gpu["W" + str(l)], mem_access_gpu["dW" + str(l)], learning_rate)
        update_weights_gpu(mem_access_gpu["b" + str(l)], mem_access_gpu["db" + str(l)], learning_rate)

def compute_cost(Y, AL):
    m = Y[2]
    if Y[1] == 1:
        cost = cost_gpu(Y[0], AL[0], AL[2])
    else:
        cost = -(1./m) * np.sum(Y * np.log(AL))

    c = np.empty(AL[2],dtype=np.float32)
    cuda.memcpy_dtoh(c, cost[0])
    
    return c

def forward_propagation(X, activation, mem_access_gpu, L):
    L = L - 1
    for l in range(1, L): 
        # start_time_gpu = time.time()
        w_ref = mem_access_gpu['W' + str(l)]
        a_ref = mem_access_gpu['A' + str(l - 1)]
        b_ref = mem_access_gpu['b' + str(l)]
        mem_access_gpu['Z' + str(l)] = forward_propagation_gpu(w = w_ref[0], a = a_ref[0], b = b_ref[0], w_rows = w_ref[1], w_cols = w_ref[2], a_cols = a_ref[2])
        # end_time_gpu = time.time()
        # time_gpu = end_time_gpu - start_time_gpu
        # print(f"time gpu: {time_gpu}")
        z_ref = mem_access_gpu['Z' + str(l)]
        if activation == 'tanh': 
            mem_access_gpu['A' + str(l)] = tanh_gpu(z_ref[0], z_rows=z_ref[1], z_cols=z_ref[2])
        else:
            mem_access_gpu['A' + str(l)] = relu_gpu(z_ref[0], z_rows=z_ref[1], z_cols=z_ref[2])

    w_ref = mem_access_gpu['W' + str(L)]
    a_ref = mem_access_gpu['A' + str(L - 1)]
    b_ref = mem_access_gpu['b' + str(L)]

    mem_access_gpu['Z' + str(L)] = forward_propagation_gpu(w = w_ref[0], a = a_ref[0], b = b_ref[0], w_rows = w_ref[1], w_cols = w_ref[2], a_cols = a_ref[2])
    z_ref = mem_access_gpu['Z' + str(L)]
    if z_ref[1] == 1:
        mem_access_gpu['A' + str(L)] = sigmoid_gpu(z_ref[0], z_rows=z_ref[1], z_cols=z_ref[2])
    else:
        mem_access_gpu['A' + str(L)] = softmax_gpu(z_ref[0], z_rows=z_ref[1], z_cols=z_ref[2])
        

if __name__ == '__main__':

    # python pycuda.py X_train.csv Y_train.csv 65536 65536 65536 1
    X_train = sys.argv[1]
    Y_train = sys.argv[2]
    X_train = np.loadtxt(X_train, delimiter = ',').astype(np.float32) / 255.0
    X_train = X_train.T
    Y_train = np.loadtxt(Y_train, delimiter = ',').astype(np.float32).reshape(1, -1)
    Y_train = Y_train.astype(np.float32)
    layer_dim = [X_train.shape[0]]
    for i in range(3, len(sys.argv)):
        layer_dim.append(int(sys.argv[i]))
    # layer_dim.append(Y_train.shape[0])
    # X_test = np.loadtxt('dataset/cat_test_x.csv', delimiter = ',')/255.0
    # Y_test = np.loadtxt('dataset/cat_test_y.csv', delimiter = ',').reshape(1, X_test.shape[1])
    lr = 0.01
    iters = 300
    model(X_train, Y_train, layer_dim, lr, activation = 'relu', num_iteration = iters)
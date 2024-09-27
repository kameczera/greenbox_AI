import time
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import os
import sys
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler as compiler
import numpy as np

def model(X, Y, layers_dim, learning_rate, activation = 'relu', num_iteration = 100):
    parameters = intialize_parameters(layers_dim)

    for i in range(0, num_iteration):
        AL, forward_cache = forward_propagation(X, parameters, activation)
        cost = compute_cost(AL, Y)
        grads = backward_propagation(AL, Y, parameters, forward_cache, activation)
        parameters = update_parameters(parameters, grads, learning_rate)

    if i % (num_iteration / 10) == 0:
        print("\niter:{} \t cost: {} \t train_acc:{} \t test_acc:{}".format(i, np.round(cost, 2), accuracy(X_train, Y_train, parameters, activation), accuracy(X_test, Y_test, parameters, activation)))
    if i % ((num_iteration / 10) / 10) == 0:
        print("==", end = '')

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

def accuracy(X, Y, parameters, activation):
    m =  Y.shape[1]

    preds, _ = forward_propagation(X, parameters, activation)
    if Y.shape[0] == 1:
        preds = np.array(preds > 0.5, dtype = 'float')
    else:
        Y = np.argmax(Y, 0)
        preds = np.argmax(preds, axis = 0)

    return np.round(np.sum(Y == preds)/m, 2) 

def intialize_parameters(layers_dim):
    L = len(layers_dim) - 1
    parameters = {}

    for l in range(1,L + 1):
        W_init = np.random.randn(layers_dim[l], layers_dim[l - 1]).astype(np.float32)
        W_scaled = W_init / np.sqrt(layers_dim[l - 1]).astype(np.float32)
        parameters["W" + str(l)] = W_scaled
        parameters["b" + str(l)] = np.zeros((layers_dim[l], 1)).astype(np.float32) # inicializacao dos bias para cada neuronio da camada
    print(parameters['W1'].dtype)
    return parameters

def backward_propagation(AL, Y, parameters, forward_cache, activation = 'relu'):
    grads = {}
    L = len(parameters)//2
    m = Y.shape[1]

    grads["dZ" + str(L)] = AL - Y
    grads["dw" + str(L)] = (1./m) * np.dot(grads["dZ" + str(L)], forward_cache["A" + str(L - 1)].T)
    grads["db" + str(L)] = (1./m) * np.sum(grads["dZ" + str(L)], axis = 1, keepdims = True)

    for l in reversed(range(1, L)):
        if activation == 'relu':
            grads["dZ" + str(l)] = np.dot(parameters["W" + str(l + 1)].T, grads["dZ" + str(L + 1)]) * derivative_relu(forward_cache["A" + str(l)])
        else:
            grads["dw" + str(l)] = (1./m) * np.dot(grads["dZ" + str(L)], forward_cache["A" + str(L - 1)].T)
            grads["db" + str(l)] = (1./m) * np.sum(grads["dZ" + str(L)], axis = 1, keepdims = True)

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
    
    return parameters

def compute_cost(AL, Y):
    m = Y.shape[1]
    
    if Y.shape[0] == 1:
        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    else:
        cost = -(1./m) * np.sum(Y * np.log(AL))
        
    cost = np.squeeze(cost)
    
    return cost

kernel_code = """
extern "C" __global__ void forward_propagation(float *a, float *w, float *b, float *z, int w_rows, int w_cols, int a_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < w_rows && col < a_cols) {
        float value = 0.0;
        for (int i = 0; i < w_cols; i++) {
            value += w[row * w_cols + i] * a[i * a_cols + col]; 
        }
        z[row * a_cols + col] = value + b[row];
    }
}
"""

mod = compiler.SourceModule(kernel_code)
forward_propagation_gpu = mod.get_function("forward_propagation")

def forward_propagation(X, parameters, activation):
   
    forward_cache = {}
    L = len(parameters) // 2
    
    forward_cache['A0'] = X
    for l in range(1, L):
        W = parameters['W' + str(l)]
        A_prev = forward_cache['A' + str(l-1)]
        b = parameters['b' + str(l)]
        
        # Dimensions
        w_rows, w_cols = W.shape
        a_rows, a_cols = A_prev.shape
        
        start_time_gpu = time.time()
        # Allocate GPU memory
        # Allocate memory on device
        w_gpu = cuda.mem_alloc(W.nbytes)
        a_gpu = cuda.mem_alloc(A_prev.nbytes)
        b_gpu = cuda.mem_alloc(b.size * np.float32().nbytes)
        z_gpu = cuda.mem_alloc(w_rows * a_cols * np.float32().nbytes)

        # Copy matrices to device memory
        cuda.memcpy_htod(w_gpu, W)
        cuda.memcpy_htod(a_gpu, A_prev)
        cuda.memcpy_htod(b_gpu, b.flatten())
        
        # Define block and grid sizes
        block_dim = (16, 16, 1)  # 16 x 16
        grid_dim = (int(np.ceil(a_cols / block_dim[0])), int(np.ceil(w_rows / block_dim[1])), 1)
        
        # Run the CUDA kernel
        forward_propagation_gpu(a_gpu, w_gpu, b_gpu, z_gpu, np.int32(w_rows), np.int32(w_cols), np.int32(a_cols), block=block_dim, grid=grid_dim)
        cuda.Context.synchronize()
        Z = np.empty((w_rows, a_cols), dtype=np.float32)
        cuda.memcpy_dtoh(Z, z_gpu)
        end_time_gpu = time.time()
        time_gpu = end_time_gpu - start_time_gpu
        start_time_cpu = time.time()
        T = parameters['W' + str(l)].dot(forward_cache['A' + str(l-1)]) + parameters['b' + str(l)]
        end_time_cpu = time.time()
        time_cpu = end_time_cpu - start_time_cpu
        print(f"time gpu: {time_gpu}, time cpu: {time_cpu}")
        forward_cache['Z' + str(l)] = Z

        if activation == 'tanh':
            forward_cache['A' + str(l)] = tanh(forward_cache['Z' + str(l)])
        else:
            forward_cache['A' + str(l)] = relu(forward_cache['Z' + str(l)])
    
    if forward_cache['Z' + str(L)].shape[0] == 1:
        forward_cache['A' + str(L)] = sigmoid(forward_cache['Z' + str(L)])
    else :
        forward_cache['A' + str(L)] = softmax(forward_cache['Z' + str(L)])
    
    return forward_cache['A' + str(L)], forward_cache

if __name__ == '__main__':
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
    parameters = model(X_train, Y_train, layer_dim, lr, activation = 'relu', num_iteration = iters)
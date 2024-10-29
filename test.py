import time
import pycuda.autoinit
from pycuda import driver, compiler, gpuarray, tools
import sys
import pycuda.driver as cuda
import numpy as np
from cuda_functions import forward_propagation_gpu, tanh_gpu, relu_gpu, sigmoid_gpu, cost_gpu, subtract_gpu, compute_dW, compute_db, compute_dZ_gpu, update_weights_gpu

def model(X_train, Y_train, layers_dim, learning_rate, activation='relu', num_iteration=100):
    L = len(layers_dim)
    mem_access_gpu = initialize_parameters(layers_dim, X_train, Y_train, L)
    
    for i in range(num_iteration):
        # Temporizadores para cada passo
        start_time = time.time()
        forward_start = time.time()
        forward_propagation(X_train, activation, mem_access_gpu, L)
        forward_end = time.time()

        cost_start = time.time()
        cost = compute_cost(mem_access_gpu["Y"], mem_access_gpu["A" + str(L - 1)])
        cost_end = time.time()

        backward_start = time.time()
        backward_propagation(mem_access_gpu["A" + str(L - 1)], mem_access_gpu["Y"], mem_access_gpu, L, activation)
        backward_end = time.time()

        update_start = time.time()
        update_parameters(mem_access_gpu, learning_rate, L)
        update_end = time.time()

        # Calculando o tempo de execução de cada etapa
        forward_time = forward_end - forward_start
        cost_time = cost_end - cost_start
        backward_time = backward_end - backward_start
        update_time = update_end - update_start
        total_time = time.time() - start_time

        if i % (num_iteration / 10) == 0:
            train_acc = accuracy(X_train, Y_train, mem_access_gpu, activation, L)
            print(f"\niter:{i} \t cost: {np.round(cost, 2)} \t train_acc: {train_acc}")
            print(f"Tempos - Forward: {forward_time:.4f}s, Cost: {cost_time:.4f}s, Backward: {backward_time:.4f}s, Update: {update_time:.4f}s, Total: {total_time:.4f}s")

            # Calculando a acurácia do conjunto de teste (comentado para não usar nesta versão)
            # test_acc = accuracy(X_test, Y_test, mem_access_gpu, activation, L)
        
        if i % ((num_iteration / 10) / 10) == 0:
            print("==", end='')
    

def accuracy(X, Y, mem_access_gpu, activation, L):
    m =  Y.shape[1]
    
    forward_propagation(X, activation, mem_access_gpu, L)
    results = np.empty((mem_access_gpu["A" + str(L - 1)][1], mem_access_gpu["A" + str(L - 1)][2]), dtype=np.float32)
    cuda.memcpy_dtoh(results, mem_access_gpu["A" + str(L - 1)][0])
    results = np.array(results > 0.5, dtype = 'float')

    return np.round(np.sum(Y == results)/m, 2) 

def initialize_parameters(layers_dim, X, Y, L):
    mem_access_gpu = {}

    a0 = cuda.mem_alloc(X.shape[0] * X.shape[1] * np.float32().nbytes)
    cuda.memcpy_htod(a0, X)

    Y_gpu = cuda.mem_alloc(Y.shape[0] * Y.shape[1] * np.float32().nbytes)
    cuda.memcpy_htod(Y_gpu, Y)
    
    mem_access_gpu['A0'] = (a0, X.shape[0], X.shape[1])
    mem_access_gpu['Y'] = (Y_gpu, Y.shape[0], Y.shape[1])
    
    # mem_access_gpu['A0'] = (gpuarray.to_gpu(X), X.shape[0], X.shape[1])
    # mem_access_gpu['Y'] = (gpuarray.to_gpu(Y), Y.shape[0], Y.shape[1])

    for l in range(1, L):
        W_init = np.random.randn(layers_dim[l], layers_dim[l - 1]).astype(np.float32)
        W_scaled = W_init / np.sqrt(layers_dim[l - 1]).astype(np.float32)
        w_rows, w_cols = W_scaled.shape
        w_gpu = cuda.mem_alloc(w_rows * w_cols * np.float32().nbytes)
        cuda.memcpy_htod(w_gpu, W_scaled)
        mem_access_gpu["W" + str(l)] = (w_gpu, w_rows, w_cols)

        b_gpu = cuda.mem_alloc(layers_dim[l] * 1 * np.float32().nbytes)
        mem_access_gpu["b" + str(l)] = (b_gpu, layers_dim[l], 1)
    
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
    L = L - 1
    for l in range(L):
        update_weights_gpu(mem_access_gpu["W" + str(l + 1)], mem_access_gpu["dW" + str(l + 1)], learning_rate)
        update_weights_gpu(mem_access_gpu["b" + str(l + 1)], mem_access_gpu["db" + str(l + 1)], learning_rate)

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
        w_ref = mem_access_gpu['W' + str(l)]
        a_ref = mem_access_gpu['A' + str(l - 1)]
        b_ref = mem_access_gpu['b' + str(l)]
        if f'Z{l}' not in mem_access_gpu:
            mem_access_gpu['Z' + str(l)] = forward_propagation_gpu(w = w_ref[0], a = a_ref[0], b = b_ref[0], w_rows = w_ref[1], w_cols = w_ref[2], a_cols = a_ref[2])
        else:
            mem_access_gpu['Z' + str(l)] = forward_propagation_gpu(w = w_ref[0], a = a_ref[0], b = b_ref[0], w_rows = w_ref[1], w_cols = w_ref[2], a_cols = a_ref[2], z = mem_access_gpu['Z' + str(l)][0])

        z_ref = mem_access_gpu['Z' + str(l)]
        if f'A{l}' not in mem_access_gpu:
            mem_access_gpu['A' + str(l)] = relu_gpu(z_ref[0], z_rows=z_ref[1], z_cols=z_ref[2])
        else:
            mem_access_gpu['A' + str(l)] = relu_gpu(z_ref[0], z_rows=z_ref[1], z_cols=z_ref[2], a = mem_access_gpu['A' + str(l)][0])
    
    w_ref = mem_access_gpu['W' + str(L)]
    a_ref = mem_access_gpu['A' + str(L - 1)]
    b_ref = mem_access_gpu['b' + str(L)]
    if f'Z{L}' not in mem_access_gpu:
        mem_access_gpu['Z' + str(L)] = forward_propagation_gpu(w = w_ref[0], a = a_ref[0], b = b_ref[0], w_rows = w_ref[1], w_cols = w_ref[2], a_cols = a_ref[2])
    else:
        mem_access_gpu['Z' + str(L)] = forward_propagation_gpu(w = w_ref[0], a = a_ref[0], b = b_ref[0], w_rows = w_ref[1], w_cols = w_ref[2], a_cols = a_ref[2], z = mem_access_gpu['Z' + str(l)][0])

    z_ref = mem_access_gpu['Z' + str(L)]
    mem_access_gpu['A' + str(L)] = sigmoid_gpu(z_ref[0], z_rows=z_ref[1], z_cols=z_ref[2])
        

if __name__ == '__main__':
    device = cuda.Device(0)
    print("Dimensão máxima (x, y, z) de um bloco:", 
      device.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_X),
      device.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_Y),
      device.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_Z))
    print("Dimensão máxima (x, y, z) de uma grade:", 
        device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_X),
        device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_Y),
        device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_Z))
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

    start_time = time.time()

    model(X_train, Y_train, layer_dim, lr, activation = 'relu', num_iteration = iters)

    end_time = time.time()

    execution_time = end_time - start_time

    print(f"Tempo de execução: {execution_time} segundos")
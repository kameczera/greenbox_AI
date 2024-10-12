import pycuda.compiler as compiler
import pycuda.driver as cuda
import numpy as np


kernel_code = """
extern "C" __global__ void matrix_mul(float *w, float *a, float *b, float *z, int w_rows, int w_cols, int a_cols) {
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

extern "C" __global__ void tanh_calc(float *z, float *a, int n_rows, int n_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n_rows && col < n_cols) {
        float e_pos = exp(z[row * n_cols + col]);  // e^x
        float e_neg = exp(-z[row * n_cols + col]); // e^-x
        a[row * n_cols + col] = (e_pos - e_neg) / (e_pos + e_neg);
    }
}

extern "C" __global__ void sigmoid_calc(float *z, float *a, int n_rows, int n_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n_rows && col < n_cols) {
        a[row * n_cols + col] = 1.0 / (1.0 + exp(-z[row * n_cols + col]));
    }
}

extern "C" __global__ void relu_calc(float *z, float *a, int n_rows, int n_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n_rows && col < n_cols) {
        a[row * n_cols + col] = fmaxf(0.0, z[row * n_cols + col]);  // ReLU: max(0, input[idx])
    }
}

extern "C" __global__ void cost_calc(float *Y, float *AL, float *cost, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < m) {
        float log_AL = logf(AL[idx]);
        float log_1_minus_AL = logf(1.0f - AL[idx]);
        
        cost[idx] = -(Y[idx] * log_AL + (1.0f - Y[idx]) * log_1_minus_AL);
    }
}

extern  "C" __global__ void subtraction(float* AL, float *Y, float *dZ, int cols){ 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < cols) {
        dZ[idx] = AL[idx] - Y[idx];
    }
}

extern "C" __global__ void compute_dW(float *dZ, float *A_prev, float *dW, int scale, int dZ_rows, int dZ_cols, int A_prev_rows, int A_prev_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < dZ_rows && col < A_prev_rows) {
        float value = 0.0;
        for (int i = 0; i < dZ_cols; i++) {
            value += dZ[row * dZ_cols + i] * A_prev[col * A_prev_cols + i];  // A_prev transposed
        }
        dW[row * A_prev_rows + col] = value / scale;
    }
}

extern "C" __global__ void compute_db(float *dZ, float *result, int rows, int cols, float scale) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        float sum = 0.0;
        for (int i = 0; i < cols; i++) {
            sum += dZ[row * cols + i]; // Sum over the columns for this row
        }
        result[row] = sum / scale; // Store the scaled sum in the result
    }
}

extern "C" __global__ void compute_dZ(float *W, float *dZ_next, float *A, float *dZ, int w_rows, int w_cols, int m) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < w_rows) {
        float sum = 0.0;
        for (int i = 0; i < w_cols; i++) {
            sum += W[row * w_cols + i] * dZ_next[i];
        }

        // Step 2: Apply the derivative of ReLU
        dZ[row] = sum * (A[row] > 0 ? 1.0f : 0.0f);
    }
}

extern "C" __global__ void update_weights(float *W, float *dW, float learning_rate, int w_rows, int w_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = row * w_cols + col;

    if (row < w_rows && col < w_cols) {
        W[idx] = W[idx] - learning_rate * dW[idx];  // Update each weight element-wise
    }
}
"""
mod = compiler.SourceModule(kernel_code)

matrix_mul_gpu = mod.get_function("matrix_mul")
tanh_calc_gpu = mod.get_function("tanh_calc")
relu_calc_gpu = mod.get_function("relu_calc")
sigmoid_calc_gpu = mod.get_function("sigmoid_calc")
cost_calc_gpu = mod.get_function("cost_calc")
subtract_calc_gpu = mod.get_function("subtraction")
compute_dW_gpu = mod.get_function("compute_dW")
db_calc_gpu = mod.get_function("compute_db")
dZ_calc_gpu = mod.get_function("compute_dZ")
update_weights = mod.get_function("update_weights")

def update_weights_gpu(w, dW, learning_rate):
    block_dim = (16, 16, 1)  # 16x16 threads per block
    grid_dim = (int(np.ceil(w[1] / block_dim[0])), int(np.ceil(w[2] / block_dim[1])), 1)

    update_weights(w[0], dW[0], np.float32(learning_rate), np.int32(w[1]), np.int32(w[2]), block=block_dim, grid=grid_dim)

def compute_dZ_gpu(w, dZ_next, A):
    dZ = cuda.mem_alloc(w[1] * np.float32().nbytes)
    block_dim = (256, 1, 1)
    grid_dim = (int(np.ceil(w[1] / block_dim[0])), 1, 1)

    dZ_calc_gpu(w[0], dZ_next[0], A[0], dZ, np.int32(w[1]), np.int32(w[2]), block=block_dim, grid=grid_dim)

    return (dZ, w[1], 1)


def forward_propagation_gpu(w, a, b, w_rows, w_cols, a_cols):
    z = cuda.mem_alloc(w_rows * a_cols * np.float32().nbytes)

    block_dim = (16, 16, 1)  # 16 x 16
    grid_dim = (int(np.ceil(a_cols / block_dim[0])), int(np.ceil(w_rows / block_dim[1])), 1)
    matrix_mul_gpu(w, a, b, z, np.int32(w_rows), np.int32(w_cols), np.int32(a_cols), block=block_dim, grid=grid_dim)
    cuda.Context.synchronize()

    return (z, w_rows, a_cols)

# compute_dW(float *dZ, float *A_prev, float *dW, int dZ_rows, int dZ_cols, int A_prev_rows, int A_prev_cols, float scale) 
def compute_dW(dZ, A, scale):
    dw = cuda.mem_alloc(dZ[1] * A[1] * np.float32().nbytes)

    block_dim = (16, 16, 1)  # 16 x 16
    grid_dim = (int(np.ceil(dZ[1] / block_dim[0])), int(np.ceil(A[1] / block_dim[1])), 1)
    compute_dW_gpu(dZ[0], A[0], dw, np.int32(scale), np.int32(dZ[1]), np.int32(dZ[2]), np.int32(A[1]), np.int32(A[2]), block=block_dim, grid=grid_dim)
    cuda.Context.synchronize()

    return (dw, dZ[1], A[2])

def compute_db(dZ, scale):
    db = cuda.mem_alloc(dZ[2] * np.float32().nbytes)

    block_dim = (256, 1, 1)
    grid_dim = (int(np.ceil(dZ[1] / block_dim[0])), 1, 1)
    
    db_calc_gpu(dZ[0], db, np.int32(dZ[1]), np.int32(dZ[2]), np.float32(scale), block=block_dim, grid=grid_dim)
    cuda.Context.synchronize()
    return (db, dZ[2], 1)

def subtract_gpu(AL, Y, Y_cols):
    dZ = cuda.mem_alloc(Y_cols * np.float32().nbytes)

    block_dim = (16, 16, 1)  # 16 x 16
    grid_dim = (int(np.ceil(Y_cols / block_dim[0])), 1, 1)
    subtract_calc_gpu(AL, Y, dZ, np.int32(Y_cols), block=block_dim, grid=grid_dim)
    cuda.Context.synchronize()

    return (dZ, 1, Y_cols)

def cost_gpu(Y, AL, AL_cols):
    cost = cuda.mem_alloc(AL_cols * np.float32().nbytes)

    block_dim = (16, 16, 1)  # 16 x 16
    grid_dim = (int(np.ceil(AL_cols / block_dim[0])), 1, 1)
    cost_calc_gpu(Y, AL, cost, np.int32(AL_cols), block=block_dim, grid=grid_dim)
    cuda.Context.synchronize()
    return (cost, 1, AL_cols)

def tanh_gpu(z, z_rows, z_cols):
    a = cuda.mem_alloc(z_rows * z_cols * np.float32().nbytes)

    block_dim = (16, 16, 1)  # 16 x 16
    grid_dim = (int(np.ceil(z_cols / block_dim[0])), int(np.ceil(z_rows / block_dim[1])), 1)
    tanh_calc_gpu(z, a, np.int32(z_rows), np.int32(z_cols), block=block_dim, grid=grid_dim)
    cuda.Context.synchronize()

    return (a, z_rows, z_cols)

def relu_gpu(z, z_rows, z_cols):
    a = cuda.mem_alloc(z_rows * z_cols * np.float32().nbytes)

    block_dim = (16, 16, 1)  # 16 x 16
    grid_dim = (int(np.ceil(z_cols / block_dim[0])), int(np.ceil(z_rows / block_dim[1])), 1)
    relu_calc_gpu(z, a, np.int32(z_rows), np.int32(z_cols), block=block_dim, grid=grid_dim)
    cuda.Context.synchronize()

    return (a, z_rows, z_cols)

def sigmoid_gpu(z, z_rows, z_cols):
    a = cuda.mem_alloc(z_rows * z_cols * np.float32().nbytes)

    block_dim = (16, 16, 1)  # 16 x 16
    grid_dim = (int(np.ceil(z_cols / block_dim[0])), int(np.ceil(z_rows / block_dim[1])), 1)
    sigmoid_calc_gpu(z, a, np.int32(z_rows), np.int32(z_cols), block=block_dim, grid=grid_dim)
    cuda.Context.synchronize()

    return (a, z_rows, z_cols)
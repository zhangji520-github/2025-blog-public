// BEGIN block_reduce
__device__ float warp_reduce(float value, int use_max) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffffu, value, offset);
        value = use_max ? fmaxf(value, other) : value + other;
    }
    return value; // lane 0 holds this Warp's result
}

// One-dimensional block, B is a multiple of 32, 32 <= B <= 1024.
// Every thread in the block must call this function.
__device__ float block_reduce(float value, int use_max) {
    __shared__ float partial[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int warps = blockDim.x / 32;
    float neutral = use_max ? -3.402823466e38f : 0.0f;
    value = warp_reduce(value, use_max);
    if (lane == 0) partial[warp] = value;
    __syncthreads();

    if (warp == 0) {
        float v = lane < warps ? partial[lane] : neutral;
        v = warp_reduce(v, use_max);
        if (lane == 0) partial[0] = v;
    }
    __syncthreads();
    float result = partial[0];
    __syncthreads(); // all readers finish before the next reuse
    return result;
}
// END block_reduce

// BEGIN softmax_rows
extern "C" __global__ void softmax_rows(
    const float* input, float* output, int N, int C) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= N) return; // uniform decision for the entire block
    const float* x = input + row * C;
    float* y = output + row * C;

    float local_max = -3.402823466e38f;
    for (int j = tid; j < C; j += blockDim.x)
        local_max = fmaxf(local_max, x[j]);
    float maximum = block_reduce(local_max, 1);

    float local_sum = 0.0f;
    for (int j = tid; j < C; j += blockDim.x) {
        float e = expf(x[j] - maximum);
        y[j] = e;
        local_sum += e;
    }
    float denominator = block_reduce(local_sum, 0);
    for (int j = tid; j < C; j += blockDim.x)
        y[j] /= denominator;
}
// END softmax_rows

// BEGIN reduce_sum
extern "C" __global__ void reduce_sum(
    const float* input, float* partial, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    float local = 0.0f;
    for (; i < count; i += step) local += input[i];
    float total = block_reduce(local, 0);
    if (threadIdx.x == 0) partial[blockIdx.x] = total;
}
// END reduce_sum

// BEGIN matmul_tiled
extern "C" __global__ void matmul_tiled(
    const float* A, const float* B, float* C, int M, int N, int K) {
    const int T = 16;
    __shared__ float As[16][16], Bs[16][16];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T + ty;
    int col = blockIdx.x * T + tx;
    float acc = 0.0f;
    for (int base = 0; base < K; base += T) {
        As[ty][tx] = row < M && base + tx < K
                  ? A[row * K + base + tx] : 0.0f;
        Bs[ty][tx] = base + ty < K && col < N
                  ? B[(base + ty) * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < T; ++k)
            acc = fmaf(As[ty][k], Bs[k][tx], acc);
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
// END matmul_tiled

// BEGIN matmul_thread_tile
extern "C" __global__ void matmul_thread_tile(
    const float* A, const float* B, float* C, int M, int N, int K) {
    const int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
    __shared__ float As[64][16], Bs[16][64];
    int tid = threadIdx.y * 16 + threadIdx.x;
    int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
    int local_row = threadIdx.y * TM, local_col = threadIdx.x * TN;
    float acc[4][4] = {};

    for (int base = 0; base < K; base += BK) {
        for (int j = tid; j < BM * BK; j += 256) {
            int m = j / BK, k = j % BK;
            As[m][k] = block_row + m < M && base + k < K
                       ? A[(block_row + m) * K + base + k] : 0.0f;
        }
        for (int j = tid; j < BK * BN; j += 256) {
            int k = j / BN, n = j % BN;
            Bs[k][n] = base + k < K && block_col + n < N
                       ? B[(base + k) * N + block_col + n] : 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < BK; ++k) {
            float a[4], b[4];
            for (int m = 0; m < TM; ++m) a[m] = As[local_row + m][k];
            for (int n = 0; n < TN; ++n) b[n] = Bs[k][local_col + n];
            for (int m = 0; m < TM; ++m)
                for (int n = 0; n < TN; ++n)
                    acc[m][n] = fmaf(a[m], b[n], acc[m][n]);
        }
        __syncthreads();
    }
    for (int m = 0; m < TM; ++m)
        for (int n = 0; n < TN; ++n) {
            int row = block_row + local_row + m;
            int col = block_col + local_col + n;
            if (row < M && col < N) C[row * N + col] = acc[m][n];
        }
}
// END matmul_thread_tile

// BEGIN vector_transpose_tile
// One 64x8 complete tile, 128 threads, input is 16-byte aligned.
extern "C" __global__ void vector_transpose_tile(
    const float* input, float* output) {
    __shared__ float AsT[8][64];
    int tid = threadIdx.x;
    int row = tid / 2, col = (tid % 2) * 4;
    float4 v = *reinterpret_cast<const float4*>(input + row * 8 + col);
    AsT[col + 0][row] = v.x;
    AsT[col + 1][row] = v.y;
    AsT[col + 2][row] = v.z;
    AsT[col + 3][row] = v.w;
    __syncthreads();
    for (int j = tid; j < 8 * 64; j += 128)
        output[j] = AsT[j / 64][j % 64];
}
// END vector_transpose_tile

// BEGIN matmul_double_buffer
extern "C" __global__ void matmul_double_buffer(
    const float* A, const float* B, float* C, int M, int N, int K) {
    const int T = 16;
    __shared__ float As[2][16][16], Bs[2][16][16];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    As[0][ty][tx] = row < M && tx < K ? A[row * K + tx] : 0.0f;
    Bs[0][ty][tx] = ty < K && col < N ? B[ty * N + col] : 0.0f;
    __syncthreads();
    int current = 0;
    float acc = 0.0f;

    for (int base = 0; base < K; base += T) {
        int next_base = base + T;
        float next_a = row < M && next_base + tx < K
                       ? A[row * K + next_base + tx] : 0.0f;
        float next_b = next_base + ty < K && col < N
                       ? B[(next_base + ty) * N + col] : 0.0f;
        for (int k = 0; k < T; ++k)
            acc = fmaf(As[current][ty][k], Bs[current][k][tx], acc);
        __syncthreads();
        int next = current ^ 1;
        As[next][ty][tx] = next_a;
        Bs[next][ty][tx] = next_b;
        __syncthreads();
        current = next;
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
// END matmul_double_buffer

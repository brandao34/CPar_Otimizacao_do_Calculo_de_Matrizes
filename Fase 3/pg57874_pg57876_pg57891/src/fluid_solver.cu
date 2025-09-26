#include <cuda_runtime.h>
#include <omp.h>
#include <cmath>
#include <iostream>
#include "fluid_solver.h"

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x)      \
    {                    \
        float *tmp = x0; \
        x0 = x;          \
        x = tmp;         \
    }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define LINEARSOLVERTIMES 20


// ? ################## ADD_SOURCE ####################################

// Add sources (density or velocity)
__global__ void add_source_cuda(int size, float *x, float *s, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && s[idx] != 0) { // Evitar cálculos desnecessários
        x[idx] += dt * s[idx];
    }
}

void add_source(int M, int N, int O, float *x, float *s, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2);
    int blockSize = 1024;
    int nBlocks = (size + blockSize - 1) / blockSize;

    add_source_cuda<<<nBlocks, blockSize>>>(size, x, s, dt);
}

// ? ################## SET_BND ####################################


__global__ void set_bnd_kernel(int M, int N, int O, int b, float *x) {
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;

    // Configurar as condições de contorno em paralelo
    if (i <= O && j <= N) {
        x[IX(i, j, 0)] = (b == 3) ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
        x[IX(i, j, O + 1)] = (b == 3) ? -x[IX(i, j, O)] : x[IX(i, j, O)];

        x[IX(0, i, j)] = (b == 1) ? -x[IX(1, i, j)] : x[IX(1, i, j)];
        x[IX(M + 1, i, j)] = (b == 1) ? -x[IX(M, i, j)] : x[IX(M, i, j)];

        x[IX(i, 0, j)] = (b == 2) ? -x[IX(i, 1, j)] : x[IX(i, 1, j)];
        x[IX(i, N + 1, j)] = (b == 2) ? -x[IX(i, N, j)] : x[IX(i, N, j)];
    }
}

// Kernel for corners
__global__ void set_bnd_corners(int M, int N, int O, float *x) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);

    x[IX(M + 1, 0, 0)] =
        0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);

    x[IX(0, N + 1, 0)] =
        0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);

    x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] +
                                      x[IX(M + 1, N + 1, 1)]);
}

void set_bnd(int M, int N, int O, int b, float *x) {
    dim3 blockDim(32, 32);

    dim3 gridDim((O + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    // Execução do kernel
    set_bnd_kernel<<<gridDim, blockDim>>>(M, N, O, b, x);

    // Corners
    set_bnd_corners<<<1, 1>>>(M, N, O, x);
}

// ? ################## LIN_SOLVE ####################################



__global__ void black_phase_kernel(int M, int N, int O, float *x, const float *x0, float a, float c, float *maxChange) {
    float local_max = 0;
    
    // Cálculo dos índices
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

    // Processamento principal
    if (i <= O && j <= N && k <= M && (k + j + 1) % 2 == i % 2) {
        int index = IX(i, j, k);
        float old_x = x[index];

        x[index] = (x0[index] + 
                   a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                        x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                        x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) / c;

        local_max = fabsf(x[index] - old_x);
    }

__shared__ float sdata[1024]; // Tamanho do bloco
float myMax = local_max;
sdata[tid] = myMax;
__syncthreads();

// Redução em árvore binária Sequencial Adressing 
for (unsigned int s = blockDim.x * blockDim.y * blockDim.z / 2; s > 0; s >>= 1) {
   if (tid < s) {
       sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
   }
   __syncthreads();
}

if (tid == 0) {
   atomicMax((int*)maxChange, __float_as_int(sdata[0]));
}
}


__global__ void red_phase_kernel(int M, int N, int O, float *x, const float *x0, float a, float c, float *maxChange) {
    float local_max = 0;
    
    // Cálculo dos índices
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

    // Processamento principal
    if (i <= O && j <= N && k <= M && (k + j) % 2 == i % 2) {
        int index = IX(i, j, k);
        float old_x = x[index];

        x[index] = (x0[index] + 
                   a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                        x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                        x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) / c;

        local_max = fabsf(x[index] - old_x);
    }

__shared__ float sdata[1024]; // Tamanho do bloco
float myMax = local_max;
sdata[tid] = myMax;
__syncthreads();

// Redução em árvore binária Sequencial Adressing 
for (unsigned int s = blockDim.x * blockDim.y * blockDim.z / 2; s > 0; s >>= 1) {
   if (tid < s) {
       sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
   }
   __syncthreads();
}

if (tid == 0) {
   atomicMax((int*)maxChange, __float_as_int(sdata[0]));
}
}

void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a,
               float c) {
    // Allocate device memory
    float *d_maxChange;
    cudaMalloc((void **)&d_maxChange, sizeof(float));

    // Define block and grid sizes
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y,
                 (O + blockDim.z - 1) / blockDim.z);

    float tol = 1e-7f, maxChange;
    int l = 0;

    do {
        maxChange = 0.0f;
        cudaMemset(d_maxChange, 0, sizeof(float));
        // Launch kernel for red points

        red_phase_kernel<<<gridDim, blockDim>>>(M, N, O, x, x0, a, c, d_maxChange);

        // Launch kernel for black points
        black_phase_kernel<<<gridDim, blockDim>>>(M, N, O, x, x0, a, c, d_maxChange);

        set_bnd(M, N, O, b, x);

        // Copy maxChange back to host
        cudaMemcpy(&maxChange, d_maxChange, sizeof(float),
                   cudaMemcpyDeviceToHost);

    } while (maxChange > tol && ++l < 20);


    cudaFree(d_maxChange);
}

// Diffusion step (uses implicit method)
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff,
             float dt) {
    int max = MAX(MAX(M, N), O);
    float a = dt * diff * max * max;
    lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

// ? ################## ADVECT ####################################


// Advection step (uses velocity field to move quantities)
__global__ void advect_kernel(int M, int N, int O, int b, float *d, const float *d0, const float *u, const float *v, const float *w, float dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

    if (i <= O && j <= N && k <= M) {
        float dtX = dt * M;
        float dtY = dt * N;
        float dtZ = dt * O;

        float x = i - dtX * u[IX(i, j, k)];
        float y = j - dtY * v[IX(i, j, k)];
        float z = k - dtZ * w[IX(i, j, k)];

        // Clamp to grid boundaries
        x = max(0.5f, min(x, M + 0.5f));
        y = max(0.5f, min(y, N + 0.5f));
        z = max(0.5f, min(z, O + 0.5f));

        int i0 = static_cast<int>(x), i1 = i0 + 1;
        int j0 = static_cast<int>(y), j1 = j0 + 1;
        int k0 = static_cast<int>(z), k1 = k0 + 1;

        float s1 = x - i0, s0 = 1 - s1;
        float t1 = y - j0, t0 = 1 - t1;
        float u1 = z - k0, u0 = 1 - u1;

        d[IX(i, j, k)] =
            s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
                  t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
            s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
                  t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
    }
}

void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v,
            float *w, float dt) {
    dim3 blockDim(16, 16, 4);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y,
                 (O + blockDim.z - 1) / blockDim.z);
    advect_kernel<<<gridDim, blockDim>>>(M, N, O, b, d, d0, u, v, w, dt);

    set_bnd(M, N, O, b, d);
}


// ? ################## PROJECT ####################################

// Projection step to ensure incompressibility (make the velocity field
// divergence-free)

__global__ void compute_div_p_and_reset(float *div, float *p, const float *u, const float *v, const float *w, int M, int N, int O) {
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

    if (i <= O && j <= N && k <= M) {
        float max_dim = max(M, max(N, O));
        div[IX(i, j, k)] = -0.5f * (
            u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] +
            v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)] +
            w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]
        ) / max_dim;

        p[IX(i, j, k)] = 0.0f; // Reset p in the same kernel
    }
}

__global__ void update_velocity(float *u, float *v, float *w, const float *p, int M, int N, int O) {
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

    if (i <= O && j <= N && k <= M) {
        u[IX(i, j, k)] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
        v[IX(i, j, k)] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
        w[IX(i, j, k)] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
    }
}

void project(int M, int N, int O, float *u, float *v, float *w, float *p,
             float *div) {
    dim3 blockDim(16, 16, 4);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y,
                 (O + blockDim.z) / blockDim.z);

    compute_div_p_and_reset<<<gridDim, blockDim>>>(div, p, u, v, w, M, N, O);

    set_bnd(M, N, O, 0, div);
    set_bnd(M, N, O, 0, p);
    lin_solve(M, N, O, 0, p, div, 1, 6);

    update_velocity<<<gridDim, blockDim>>>(u, v, w, p, M, N, O);

    set_bnd(M, N, O, 1, u);
    set_bnd(M, N, O, 2, v);
    set_bnd(M, N, O, 3, w);
}

// Step function for density
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v,
               float *w, float diff, float dt) {         
    add_source(M, N, O, x, x0, dt);
    SWAP(x0, x);
    diffuse(M, N, O, 0, x, x0, diff, dt);
    SWAP(x0, x);
    advect(M, N, O, 0, x, x0, u, v, w, dt);
}

// Step function for velocity
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0,
              float *v0, float *w0, float visc, float dt) {
    add_source(M, N, O, u, u0, dt);
    add_source(M, N, O, v, v0, dt);
    add_source(M, N, O, w, w0, dt);
    SWAP(u0, u);
    diffuse(M, N, O, 1, u, u0, visc, dt);
    SWAP(v0, v);
    diffuse(M, N, O, 2, v, v0, visc, dt);
    SWAP(w0, w);
    diffuse(M, N, O, 3, w, w0, visc, dt);
    project(M, N, O, u, v, w, u0, v0);
    SWAP(u0, u);
    SWAP(v0, v);
    SWAP(w0, w);
    advect(M, N, O, 1, u, u0, u0, v0, w0, dt);
    advect(M, N, O, 2, v, v0, u0, v0, w0, dt);
    advect(M, N, O, 3, w, w0, u0, v0, w0, dt);
    project(M, N, O, u, v, w, u0, v0);
}
/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve_cuda.cuh"


/* 
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source: 
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

/*
Warp reduction function which performs max reduction for a warp.
Instructions within a warp are synchronous so don't need to call
__syncthreads()
For loop for reduction has been unrolled.
This saves work since without unrolling, all warps would execute every
iteration of the for loop and if statement
 */
template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, int tid) {
    if (blockSize >= 64) sdata[tid] = max(sdata[tid], sdata[tid+32]);
    if (blockSize >= 32) sdata[tid] = max(sdata[tid], sdata[tid+16]);
    if (blockSize >= 16) sdata[tid] = max(sdata[tid], sdata[tid+8]);
    if (blockSize >= 8) sdata[tid] = max(sdata[tid], sdata[tid+4]);
    if (blockSize >= 4) sdata[tid] = max(sdata[tid], sdata[tid+2]);
    if (blockSize >= 2) sdata[tid] = max(sdata[tid], sdata[tid+1]);
}

__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data,
    int padded_length) {


    /* Point-wise multiplication and scaling for the
    FFT'd input and impulse response. 

    Rule for multiplying two complex numbers a and b is
    real result = a.x*b.x-a.y*b.y
    imag result = a.x*b.y+a.y*b.x
    */
   
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread idx will handle the convolution of a single element
    // from the input array at each step
    cufftComplex a;
    cufftComplex b;
    cufftComplex result;
    while (i < padded_length) {
        // Load in input and impulse response
        a = raw_data[i];
        b = impulse_v[i];
        // Multiply input and resule and scaled by padded length
        result.x = (a.x*b.x-a.y*b.y) / padded_length;
        result.y = (a.x*b.y+a.y*b.x) / padded_length;
        out_data[i] = result;
        // Increment thread idx if not enough blocks
        i += blockDim.x * gridDim.x;
    }
}

template <unsigned int blockSize>
__global__ void cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* My approach for this reduction is based on the Harris Nvidia slides on optimizing
    reduction from
    https://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf
    I use a tree-based approach within each thread block. So each block will
    find the max of a partition of the input_data. They will do this by reducing
    the problem in half each step by finding the max of half of the partition and
    finding the max of the half of that and so on until the block finds the max
    of the partition. Then each block will use atomicMax to compare their found 
    max value to the global max value.
    I use shared memory so that each block can load in data from global memory
    and then used shared memory to perform the reduction.
    Optimizations:
    1. During the load from global memory to shared memory, instead of doing only
    a single load per thread we can do two loads and the first step of the reduction
    in one step while halving the number of blocks needed. So a thread will load in two 
    values from global memory that are 1 blockSize apart from each other and take their max.
    If instead during the first step we have each thread load in a value, sync_threads, and
    then have half the threads take the max of their own and another value, the other half
    of the threads will be idle. So we avoid having idle threads during the first step
    of the reduction.
    We can optimize this further by doing it in a while loop and doing as many max()'s
    as necessary if there are not enough blocks.
    2. If we use interleaved addressing for each step of the reduction then we will
    have shared memory bank conflicts since threads in a warp will be accessing
    different addresses from the same bank. To avoid this, we can do sequential adressing
    where a warp will access a sequential part of shared memory. So if the block size
    is 32, then during the first step, a thread will compare two values 16 apart from each
    other.
    3. As the steps in the reduction proceed, the number of active threads decreases.
    When we have less than or equal to 64 values to reduce, then we only have one warp left
    since we will have 32 threads in a warp compare 32 consecutive values to the next 32
    consecutive values. So we can unroll the last 6 steps of the reduction inside the function
    warpReduce(). Unrolling the loop reduces dependencies.
    Furthermore, instructions are synchonous within a warp so we don't need to call __syncthreads().
    4. To completely unroll the loop we need to know the number of iterations, which we can find
    from the block size. Since block sizes are in powers of 2 and limited to 512 threads, then
    we can unroll the loop for all 10 possible cases. We will need to use template parameters and
    specify block size as a function template parameter. This let's us adjust the code depending
    in the block size since during compile time, the compiler will evaluate if statements with
    block size.
    
    */
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x *2* (blockDim.x) + tid; 
    unsigned int gridSize = blockSize*2*gridDim.x;

    // Initialize shared mem to 0
    sdata[tid] = 0;

    // Perform two loads and the first step of the reduction as many times
    // as needed. Optimization 1.
    while (i+blockDim.x < padded_length) { 
        sdata[tid] = max(sdata[tid],max(abs(out_data[i].x), abs(out_data[i+blockDim.x].x)));
        i += gridSize; // Step by gridSize each time to maintain coalescing
    }    
    __syncthreads(); // Sync threads are loading in to shared memory

    // Unroll for loop for all possible cases of block size. Optimization 4.
    // Sync threads after step of the reduction.
    // Use sequential addressing to avoid memory bank conflicts. Optimization 2.
    if (blockSize >= 512) {if (tid < 256) { sdata[tid] = max(sdata[tid], sdata[tid+256]);} __syncthreads(); } 
    if (blockSize >= 256) {if (tid < 128) { sdata[tid] = max(sdata[tid], sdata[tid+128]);} __syncthreads(); } 
    if (blockSize >= 128) {if (tid < 64) { sdata[tid] = max(sdata[tid], sdata[tid+64]);} __syncthreads(); } 

    // We're down to a single warp and no longer need to syncthreads
    // Reduce for a single warp
    if (tid < 32) warpReduce<blockSize>(sdata, tid);

    // Use atomic max so after reductions are done, first thread of each block 
    // compares its found max value to global max
    if (tid == 0) atomicMax(max_abs_val, sdata[0]);
    

}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /*Divide all data by the value pointed to by max_abs_val. */
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    while(i < padded_length) {
        out_data[i].x /= max_abs_val[0];
        i += blockDim.x * gridDim.x;
    }
}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
        
    /* Call the element-wise product and scaling kernel. */
    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>(raw_data, impulse_v,
            out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        

    /* Call the max-finding kernel. */
    // We need to specify the correct blockSize for the kernel call
    // since it's a template parameter but it needs to be a constant. 
    // There are 10 cases of block size so we can call the kernel with
    // the correct blockSize constant for each case
    switch (threadsPerBlock) {
        case 512: cudaMaximumKernel<512><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
        case 256: cudaMaximumKernel<256><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
        case 128: cudaMaximumKernel<128><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
        case 64: cudaMaximumKernel<64><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
        case 32: cudaMaximumKernel<32><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
        case 16: cudaMaximumKernel<16><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
        case 8: cudaMaximumKernel<8><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
        case 4: cudaMaximumKernel<4><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
        case 2: cudaMaximumKernel<2><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
        case 1: cudaMaximumKernel<1><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
    }
}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* Call the division kernel. */
    cudaDivideKernel<<<blocks, threadsPerBlock>>>(out_data, max_abs_val, padded_length);
}

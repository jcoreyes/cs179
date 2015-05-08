#include <cassert>
#include <cuda_runtime.h>
#include "cluster_cuda.cuh"

// This assumes address stores the average of n elements atomically updates
// address to store the average of n + 1 elements (the n elements as well as
// val). This might be useful for updating cluster centers.
// modified from http://stackoverflow.com/a/17401122
__device__ 
float atomicUpdateAverage(float* address, int n, float val)
{
  int* address_as_i = (int*) address;
  int old = *address_as_i;
  int assumed;
  do {
    assumed = old;
    float next_val = (n * __int_as_float(assumed) + val) / (n + 1);
    old = ::atomicCAS(address_as_i, assumed,
		      __float_as_int(next_val));
  } while (assumed != old);
  return __int_as_float(old);
}

// computes the distance squared between vectors a and b where vectors have
// length size and stride stride.
__device__ 
float squared_distance(float *a, float *b, int stride, int size) {
  float dist = 0.0;
  for (int i=0; i < size; i++) {
    float diff = a[stride * i] - b[stride * i];
    dist += diff * diff;
  }
  return dist;
}

/*
 * Notationally, all matrices are column majors, so if I say that matrix Z is
 * of size m * n, then the stride in the m axis is 1. For purposes of
 * optimization (particularly coalesced accesses), you can change the format of
 * any array.
 *
 * clusters is a REVIEW_DIM * k array containing the location of each of the k
 * cluster centers.
 *
 * cluster_counts is a k element array containing how many data points are in 
 * each cluster.
 *
 * k is the number of clusters.
 *
 * data is a REVIEW_DIM * batch_size array containing the batch of reviews to
 * cluster. Note that each review is contiguous (so elements 0 through 49 are
 * review 0, ...)
 *
 * output is a batch_size array that contains the index of the cluster to which
 * each review is the closest to.
 *
 * batch_size is the number of reviews this kernel must handle.
 */
__global__
void sloppyClusterKernel(float *clusters, int *cluster_counts, int k, 
                          float *data, int *output, int batch_size) {
    unsigned int tid = threadIdx.x
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float cluster_data[];
    unsigned int tid = threadIdx.x;
    // Load cluster centers into shared memory
    while (tid < k*REVIEW_DIM) {
        sdata[tid] = clusters[tid];
        tid += blockDim.x;
    }
    __syncthreads(); 

    while (i < batch_size) {
        // Initialize min dist to max possible float
        float min_dist = 1e10;
        int closest = 0;
        // Iterate through clusters and keep track of closest center
        for (int c=0; c < k; c++) {
            // Compute distance from a review to a cluster
            float curr_dist = 0;
            for (int j=0; j < REVIEW_DIM; j++) {
                float dist = (data[i*REVIEW_DIM + j] - cluster_data[c*REVIEW_DIM + j]);
                curr_dist += dist * dist;
            }
            // Update if distance is less
            if (curr_dist < min_dist) {
                min_dist == curr_dist;
                closest = c;
            }

        }
        // Write closest center
        output[i] = closest;
        // Update average of closest_cluster
        int n = cluster_counts[closest];
        float s_n;
        for (int j=0; j < REVIEW_DIM; j++) {
            s_n = clusters[closest*REVIEW_DIM + j]; 
            clusters[closest*REVIEW_DIM + j] = (n * s_n + data[i*REVIEW_DIM+j]) / (n + 1)
        }        
        cluster_counts[closest]++;
        // Increment if not enough threads
        i += blockDim.x * gridDim.x;
    }
}


void cudaCluster(float *clusters, int *cluster_counts, int k,
		 float *data, int *output, int batch_size, 
		 cudaStream_t stream) {
  int block_size = (batch_size < 1024) ? batch_size : 1024;

  // grid_size = CEIL(batch_size / block_size)
  int grid_size = (batch_size + block_size - 1) / block_size;
  int shmem_bytes = REVIEW_DIM * k * sizeof(float);

  sloppyClusterKernel<<<
    block_size, 
    grid_size, 
    shmem_bytes, 
    stream>>>(clusters, cluster_counts, k, data, output, batch_size);
}

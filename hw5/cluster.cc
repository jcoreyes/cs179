#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <sstream>
#include <fstream>
#include <math.h>
#include <cuda_runtime.h>

#include "cluster_cuda.cuh"

using namespace std;

/*
NOTE: You can use this macro to easily check cuda error codes
and get more information.

Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                                            const char *file,
                                            int line,
                                            bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
                        cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// timing setup code
cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
            gpuErrChk(cudaEventCreate(&start));       \
            gpuErrChk(cudaEventCreate(&stop));        \
            gpuErrChk(cudaEventRecord(start));        \
        }

#define STOP_RECORD_TIMER(name) {                           \
            gpuErrChk(cudaEventRecord(stop));                     \
            gpuErrChk(cudaEventSynchronize(stop));                \
            gpuErrChk(cudaEventElapsedTime(&name, start, stop));  \
            gpuErrChk(cudaEventDestroy(start));                   \
            gpuErrChk(cudaEventDestroy(stop));                    \
    }

////////////////////////////////////////////////////////////////////////////////
// Start non boilerplate code

// Fills output with standard normal data
void gaussianFill(float *output, int size) {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (int i=0; i < size; i++) {
        output[i] += distribution(generator);
        output[i] *= distribution(generator);
        output[i] -= distribution(generator);
    }
}

// Takes a string of comma seperated floats and stores the float values into
// output. Each string should consist of REVIEW_DIM floats.
void readLSAReview(string review_str, float *output) {
    stringstream stream(review_str);
    int component_idx = 0;

    for (string component; getline(stream, component, ','); component_idx++) {
        output[component_idx] = atof(component.c_str());
    }
    assert(component_idx == REVIEW_DIM);
}

// used to pass arguments to printerCallback
struct printerArg {
    int review_idx_start;
    int batch_size;
    int *cluster_assignments;
};

// Prints out which cluster each review in a batch was assigned to.
// TODO: Call with cudaStreamAddCallback (after completing D->H memcpy)
void printerCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printerArg *arg = static_cast<printerArg *>(userData);

    for (int i=0; i < arg->batch_size; i++) {
        printf("%d: %d\n", 
             arg->review_idx_start + i, 
             arg->cluster_assignments[i]);
    }

    delete arg;
}

void cluster(int k, int batch_size) {
    // cluster centers
    float *d_clusters;

    // how many points lie in each cluster
    int *d_cluster_counts;

    // allocate memory for cluster centers and counts
    gpuErrChk(cudaMalloc(&d_clusters, k * REVIEW_DIM * sizeof(float)));
    gpuErrChk(cudaMalloc(&d_cluster_counts, k * sizeof(int)));

    // randomly initialize cluster centers
    float *clusters = new float[k * REVIEW_DIM];
    gaussianFill(clusters, k * REVIEW_DIM);
    gpuErrChk(cudaMemcpy(d_clusters, clusters, k * REVIEW_DIM * sizeof(float),
                             cudaMemcpyHostToDevice));

    // initialize cluster counts to 0
    gpuErrChk(cudaMemset(d_cluster_counts, 0, k * sizeof(int)));
    
    // TODO: allocate copy buffers and streams
    int buff_byte_size = batch_size * REVIEW_DIM * sizeof(float);
    // Allocate two each of host, dev input, and dev output buffers
    float **host_buffs = (float **) malloc(2 * sizeof(float*));
    float **dev_input_buffs = (float **) malloc(2 * sizeof(float*));
    int **dev_output_buffs = (int **) malloc(2 * sizeof(int*));
    // Allocate stream
    cudaStream_t stream[2];

    for (int i=0; i < 2; i++) {
        gpuErrChk(cudaMalloc(&dev_input_buffs[i], buff_byte_size));
        gpuErrChk(cudaMalloc(&dev_output_buffs[i], batch_size * sizeof(int)));
        //gpuErrChk(cudaMallocHost(&host_buffs[i], buff_byte_size));
        host_buffs[i] = (float*) malloc(buff_byte_size);
        gpuErrChk(cudaHostRegister(host_buffs[i], buff_byte_size, 0));
        gpuErrChk(cudaStreamCreate(&stream[i]));
    }
    
    // initialize timers
    float batch_ms = -1;
    float total_ms = -1;
    float copy1_ms = 0;
    float copy2_ms = 0;
    float cluster_ms = 0;
    // main loop to process input lines (each line corresponds to a review)
    int review_idx = 0;
    int buffer_no = 0; // What buffer we're currently using
    // testbuff.txt is the first 50000 lines of shuffled_lsa.txt
    ifstream ifs("testbuff.txt");
    stringstream testbuffer;
    testbuffer << ifs.rdbuf();

    // Start timer
    START_TIMER();

    //for (string review_str; getline(testbuffer, review_str); review_idx++) {
    for (string review_str; getline(cin, review_str); review_idx++) {
        // Load in review to appropiate host buffer
        readLSAReview(review_str, host_buffs[buffer_no] + REVIEW_DIM * (review_idx % batch_size));
        // If no more reviews but not a complete batch then adjust size
        
        if (cin.peek() == EOF && (review_idx+1) % batch_size != 0) {
            batch_size = review_idx % batch_size + 1;
            printf("Reached end of file at %d\n", review_idx);
            printf("Setting batch size to %d\n", batch_size); 
        }
        
        // If we filled a buffer or no more reviews, then begin stream
        if ((review_idx+1) % batch_size == 0 && review_idx > 0 || cin.peek() == EOF) {
            
            // Allocate printer arguments for printing cluster assignments
            struct printerArg *printer_arg = (struct printerArg *) malloc(sizeof(printerArg));
            printer_arg->review_idx_start = review_idx;
            printer_arg->batch_size = batch_size;
            printer_arg->cluster_assignments = (int*) malloc(batch_size*sizeof(int));
            START_TIMER();
            // Asynchronous copy review data from host to device
            cudaMemcpyAsync(dev_input_buffs[buffer_no], host_buffs[buffer_no], buff_byte_size,
                    cudaMemcpyHostToDevice, stream[buffer_no]);
            cudaStreamSynchronize(stream[buffer_no]);
            STOP_RECORD_TIMER(batch_ms);
            copy1_ms += batch_ms;
            printf("Host to device copy time %fms with bandwidth %f (GB/s)\n", batch_ms, buff_byte_size/batch_ms/1e6);
            // Kernel call to cluster batch of reviews
            START_TIMER();
            cudaCluster(d_clusters, d_cluster_counts, k, dev_input_buffs[buffer_no], 
                    dev_output_buffs[buffer_no], batch_size, stream[buffer_no]);
            cudaStreamSynchronize(stream[buffer_no]);
            STOP_RECORD_TIMER(batch_ms);
            cluster_ms += batch_ms;
            printf("Cluster time %fms\n", batch_ms);
            // Asynchonrous copy cluster assignments from device to host
            START_TIMER();
            cudaMemcpyAsync(printer_arg->cluster_assignments, dev_output_buffs[buffer_no], 
                    batch_size*sizeof(int), cudaMemcpyDeviceToHost, stream[buffer_no]);

            cudaStreamSynchronize(stream[buffer_no]);
            STOP_RECORD_TIMER(batch_ms);
            copy2_ms += batch_ms;
            printf("Device to host copy time %fms with bandwidth %f(GB/s)\n", batch_ms, batch_size*sizeof(int)/batch_ms/1e6);
            // Add callback for printing cluster assignments once done.
            //cudaStreamAddCallback(stream[buffer_no], printerCallback, (void*)printer_arg, 0);

            // Switch buffer number after using current one
            if (buffer_no == 1) {
                buffer_no = 0;
            }
            else {
                buffer_no = 1;
            }
        }

    }

    // wait for everything to end on GPU before final summary
    gpuErrChk(cudaDeviceSynchronize());

    // retrieve final cluster locations and counts
    int *cluster_counts = new int[k];
    gpuErrChk(cudaMemcpy(cluster_counts, d_cluster_counts, k * sizeof(int), 
                             cudaMemcpyDeviceToHost));
    gpuErrChk(cudaMemcpy(clusters, d_clusters, k * REVIEW_DIM * sizeof(int),
                             cudaMemcpyDeviceToHost));

    //STOP_RECORD_TIMER(total_ms);
    //printf("Batch size: %d with throughput %f reviews/s\n", batch_size, review_idx/total_ms *1000);
    // print cluster summaries
    
    for (int i=0; i < k; i++) {
        printf("Cluster %d, population %d\n", i, cluster_counts[i]);
        printf("[");
        for (int j=0; j < REVIEW_DIM; j++) {
            printf("%.4e,", clusters[i * REVIEW_DIM + j]);
        }
        printf("]\n\n");
    }
    
    int total_b = (int) ceil((float)review_idx / batch_size);
    printf("Batch size: %d\nTotal: %f\nD->H copy: %f\nCluster Kernel: %f\nH->D copy: %f\n",
         batch_size, (copy1_ms + cluster_ms + copy2_ms) / total_b, copy1_ms/total_b, cluster_ms/total_b, copy2_ms/total_b);
    // free cluster data
    gpuErrChk(cudaFree(d_clusters));
    gpuErrChk(cudaFree(d_cluster_counts));
    delete[] cluster_counts;
    delete[] clusters;

    for (int i=0; i < 2; i++) {
        gpuErrChk(cudaFree(dev_input_buffs[i]));
        gpuErrChk(cudaFree(dev_output_buffs[i]));
        //cudaFreeHost(host_buffs[i]);
        gpuErrChk(cudaHostUnregister(host_buffs[i]));
        free(host_buffs[i]);
        gpuErrChk(cudaStreamDestroy(stream[i]));
    }
    delete[] dev_input_buffs;
    delete[] dev_output_buffs;
    delete[] host_buffs;

}

int main() {
   // cluster(50, 2048);
    //cluster(50, 4096);
    //cluster(50, 512);
    cluster(5, 2048);
    return 0;
}

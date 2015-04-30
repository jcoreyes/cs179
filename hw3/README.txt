Based on my optimizations for the max reduction (as explained in fft_convolve_cuda.cu),
the threadsPerBlock argument must be a power of 2 and a maximum size of 512.
Refer to comments in fft_convolve_cuda.cu for description of approach and optimizations.
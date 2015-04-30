
/* 
Based off work by Nelson, et al.
Brigham Young University (2010)

Adapted by Kevin Yuh (2015)
*/


#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cufft.h>
#define PI 3.14159265358979

/* Check errors on CUDA runtime functions */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

texture<float,2,cudaReadModeElementType> texreference;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

/* Check errors on cuFFT functions */
void gpuFFTchk(int errval){
    if (errval != CUFFT_SUCCESS){
        printf("Failed FFT call, error code %d\n", errval);
    }
}

/* Check errors on CUDA kernel calls */
void checkCUDAKernelError()
{
    cudaError_t err = cudaGetLastError();
    if  (cudaSuccess != err){
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    } else {
        fprintf(stderr, "No kernel error detected\n");
    }

}

__global__ void cudaExtractReal(cufftComplex *dev_sinogram_cmplx, float *dev_sinogram, const int totalSize) {

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    while(i < totalSize) {
        dev_sinogram[i] = dev_sinogram_cmplx[i].x;
        i += blockDim.x * gridDim.x;
    }
 
}

void cudaCallExtractReal(const unsigned int nBlocks, const unsigned int threadsPerBlock,
        cufftComplex *dev_sinogram_cmplx, float *dev_sinogram_float, const int totalSize){

    cudaExtractReal<<<nBlocks, threadsPerBlock>>>( dev_sinogram_cmplx, dev_sinogram_float, totalSize);
}

__global__ void cudaFrequencyScaleKernel(cufftComplex *dev_sinogram_cmplx,
    const int sinogram_width, const int nAngles) {

    const int totalSize = nAngles * sinogram_width;

    /*Divide all data by the value pointed to by max_abs_val. */
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    // For a given angle, scaling factor is 1 - dist_from_center / (n/2)
    // = 1 - abs(n/2 - i) / (n/2) = 1 - abs(1 - 2*i/n)
    float scalingFactor = 1 - fabsf(1 - 2*(i % sinogram_width)) / sinogram_width;
    //float scalingFactor = 0.5; 
    while(i < totalSize) {
        dev_sinogram_cmplx[i].x *= scalingFactor;
        dev_sinogram_cmplx[i].y *= scalingFactor;
        i += blockDim.x * gridDim.x;
    }
}


void cudaCallFrequencyScaleKernel(const unsigned int blocks, const unsigned int threadsPerBlock,
    cufftComplex *dev_sinogram_cmplx, const int sinogram_width, const int nAngles) {

    //cudaFrequencyScaleKernel<<<blocks, threadsPerBlock>>>(dev_sinogram_cmplx, sinogram_width, nAngles);

}

__global__ void cudaBackProjection(float *output_dev, float *dev_sinogram, const int sinogram_width,
    const int nAngles, const int width, const int height, const float theta_step,const int mid_width,
    const int mid_height,const int mid_sinogram_width) {

    int x = blockIdx.x * blockDim.x + threadIdx.x; // pixel coord
    int y = blockIdx.y * blockDim.y + threadIdx.y; // pixel coord
    int x_geo = x - mid_width;
    int y_geo = y - mid_height;
    float x_i, y_i;
    int d;
    float theta, m, q;
    //for(;x < width; x += blockDim.x * gridDim.x) {
    //    for(; y < height; y += blockDim.y * gridDim.y) {
    for(int thetaNo = 0 ; thetaNo < nAngles; thetaNo++) {
        // Calculate theta based on angle number
        theta = (float)thetaNo * theta_step;

        if (theta == 0) {
            d = x_geo;
        }
        else if (theta == PI/2) {
            d = y_geo;
        }
        else if (theta == PI) {
            d = -x_geo;
        }
        else if (theta == 3*PI/2) {
            d = -y_geo;
        }
        else {
             // Calculate slope from theta
            m = -cos(theta)/sin(theta);
            q = -1.0/m;
            // Handle edge cases
            x_i = (float)(y_geo - m*x_geo) / (q - m);
            y_i = q*x_i;
            d = (int) sqrtf((x_i*x_i + y_i*y_i));

            // Use -d instead of d when x_i < 0 or if -1/m < 0 and x_i ? 0
            if (x_i < 0 || (q < 0 && x_i > 0))
                d = -d;               
        }
        //output_dev[y*width + x] += tex2D(texreference, mid_sinogram_width + d, thetaNo);
        output_dev[y*width + x] += dev_sinogram[mid_sinogram_width + d + thetaNo *sinogram_width];
    }
    //    }
    //}
}

void cudaCallBackProjection(const dim3 blocknum, const dim3 blocksize, 
    float *output_dev, float *dev_sinogram_float, 
    const int sinogram_width, const int nAngles,const int width,const int height,
    const float theta_step, const int mid_width,const int mid_height,const int mid_sinogram_width) {
    cudaBackProjection<<<blocknum, blocksize>>>(output_dev, 
        dev_sinogram_float, sinogram_width, nAngles, width, height, theta_step,
        mid_width, mid_height, mid_sinogram_width);
}



int main(int argc, char** argv){
    printf("Starting program\n");
    if (argc != 7){
        fprintf(stderr, "Incorrect number of arguments.\n\n");
        fprintf(stderr, "\nArguments: \n \
        < Sinogram filename > \n \
        < Width or height of original image, whichever is larger > \n \
        < Number of angles in sinogram >\n \
        < threads per block >\n \
        < number of blocks >\n \
        < output filename >\n");
        exit(EXIT_FAILURE);
    }

    /********** Parameters **********/

    int width = atoi(argv[2]);
    int height = width;
    int sinogram_width = (int)ceilf( height * sqrt(2) );
    int nAngles = atoi(argv[3]);

    int threadsPerBlock = atoi(argv[4]);
    int nBlocks = atoi(argv[5]);

    int sinogram_cmplx_byte_size = (sinogram_width*nAngles*sizeof(cufftComplex));
    int sinogram_byte_size = sinogram_width*nAngles*sizeof(float);
    float theta_step = PI / (float) nAngles;
    int mid_width = (int) floor((float) width / 2.0);
    int mid_height = (int) floor((float) height / 2.0);
    int mid_sinogram_width = (int) floor((float) sinogram_width/2.0);
    /********** Data storage *********/

    // GPU DATA STORAGE
    cufftComplex *dev_sinogram_cmplx;
    float *dev_sinogram_float; 
    float* output_dev;  // Image storage

    // Texture data storage
    dim3 blocknum;
    dim3 blocksize;
    cudaArray* carray;
    cudaChannelFormatDesc channel;

    // Host data storage
    cufftComplex *sinogram_host;
    size_t size_result = width*height*sizeof(float);
    float *output_host = (float *)malloc(size_result);
    float *sinogram_float = (float *)malloc(sinogram_byte_size);

    /*********** Set up IO, Read in data ************/
    printf("sinogram_width: %d, theta_step: %f, mid_width, mid_height, mid_sinogram_width: %d %d %d\n",
            sinogram_width, theta_step, mid_width, mid_height, mid_sinogram_width);
    sinogram_host = (cufftComplex *)malloc(  sinogram_width *nAngles * sizeof(cufftComplex) );

    FILE *dataFile = fopen(argv[1],"r");
    if (dataFile == NULL){
        fprintf(stderr, "Sinogram file missing\n");
        exit(EXIT_FAILURE);
    }

    FILE *outputFile = fopen(argv[6], "w");
    if (outputFile == NULL){
        fprintf(stderr, "Output file cannot be written\n");
        exit(EXIT_FAILURE);
    }

    int j, i;
    printf("Reading input sinogram file\n");
    for(i = 0; i < nAngles * sinogram_width; i++){
        fscanf(dataFile,"%f",&sinogram_host[i].x);
        sinogram_host[i].y = 0;
    }

    fclose(dataFile);
    printf("Finished reading input sinogram file\n");

    /*********** Assignment starts here *********/

    /* TODO: Allocate memory for all GPU storage above, copy input sinogram
    over to dev_sinogram_cmplx. */
    cudaMalloc((void **)&dev_sinogram_cmplx, sinogram_cmplx_byte_size);
    cudaMalloc((void **)&dev_sinogram_float, sinogram_byte_size);

    printf("Copying sinogram from host to device for high pass filter\n");
    gpuErrchk( cudaMemcpy(dev_sinogram_cmplx, sinogram_host, sinogram_cmplx_byte_size,
                 cudaMemcpyHostToDevice));

    /* TODO 1: Implement the high-pass filter:
        - Use cuFFT for the forward FFT
        - Create your own kernel for the frequency scaling.
        - Use cuFFT for the inverse FFT
        - extract real components to floats
        - Free the original sinogram (dev_sinogram_cmplx)

        Note: If you want to deal with real-to-complex and complex-to-real
        transforms in cuFFT, you'll have to slightly change our code above.
    */
   
    /* Create a cuFFT plan for the forward transform. */
    cufftHandle plan;
    int batch = nAngles; // Number of transforms to run

    cufftPlan1d(&plan, sinogram_cmplx_byte_size, CUFFT_C2C, batch);
    /* Run the forward DFT on the input signal in-place */
    cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_FORWARD);

    printf("Executing frequency scale kernel\n");
    /* Call frequency scaling kernel */
    cudaCallFrequencyScaleKernel(nBlocks, threadsPerBlock, dev_sinogram_cmplx, sinogram_width, nAngles);
    printf("Finished executing scale kernel\n");

    /* Create new cuFFT plan for backward transform. */
    //cufftPlan1d(&plan, sinogram_cmplx_byte_size, CUFFT_C2R, batch);
    /* Run backward DFT on output signal and extract real part */
    //cufftExecC2R(plan, dev_sinogram_cmplx, dev_sinogram_float);
    cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_INVERSE);

    printf("Copying sinogram data data back to host\n");
    cudaCallExtractReal(nBlocks, threadsPerBlock, dev_sinogram_cmplx, dev_sinogram_float, sinogram_width*nAngles);
    /* Copy data back to host */
    cudaMemcpy( sinogram_float, dev_sinogram_float, sinogram_byte_size, cudaMemcpyDeviceToHost);
    cudaFree(dev_sinogram_cmplx);


    /* TODO 2: Implement backprojection.
        - Allocate memory for the output image.
        - Create your own kernel to accelerate backprojection.
        - Copy the reconstructed image back to output_host.
        - Free all remaining memory on the GPU.
    */
    cudaMalloc((void **)&output_dev, size_result);
    cudaMemset(output_dev, 0, size_result);
    /* Set up texture memory */
    // Create channel to descibe data type
    channel = cudaCreateChannelDesc<float>();
    cudaMallocArray(&carray, &channel, sinogram_width, nAngles);
    // Copy sinogram from host to device
    cudaMemcpyToArray(carray, 0, 0, sinogram_float, sinogram_byte_size, cudaMemcpyHostToDevice);

    // Set texture filterm mode property to linear
    texreference.normalized = 0;
    texreference.filterMode = cudaFilterModeLinear;
    // Set texture address mode to clamp
    texreference.addressMode[0] = cudaAddressModeClamp;
    texreference.addressMode[1] = cudaAddressModeClamp;
    // Bind texture reference with cuda array
    cudaBindTextureToArray(texreference,carray);
    blocksize.x=16;
    blocksize.y=16;
    blocknum.x=(int) ceil((float)width/16);
    blocknum.y=(int) ceil((float)height/16);

    printf("Executing back projection kernel\n");
    cudaCallBackProjection(blocknum, blocksize, output_dev, 
        dev_sinogram_float, sinogram_width, nAngles, width, height,theta_step,
        mid_width, mid_height, mid_sinogram_width);
    printf("Finished executing back projection kernel\n");
    //Unbind texture reference to free resource
    cudaUnbindTexture(texreference);

    // Copy result matrix from device to host
    cudaMemcpy( output_host, output_dev, size_result, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_sinogram_float);
    cudaFree(output_dev);
    cudaFreeArray(carray);
    /* Export image data. */

    for(j = 0; j < width; j++){
        for(i = 0; i < height; i++){
            fprintf(outputFile, "%e ",output_host[j*width + i]);
        }
        fprintf(outputFile, "\n");
    }


    /* Cleanup: Free host memory, close files. */
    free(sinogram_float);
    free(sinogram_host);
    free(output_host);

    fclose(outputFile);

    return 0;
}



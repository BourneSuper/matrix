/*
 * Matrix is a PHP extension. It can do parallel computing base on CUDA.
 *
 * GitHub: https://github.com/BourneSuper/matrix
 *
 * Author: Bourne Wong <cb44606@gmail.com>
 *
 * */


#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include "math.cuh"



int getMaxThreadsPerMultiProcessor(  deviceContextStruct * deviceContextStructP ){
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties( &deviceProp, deviceContextStructP->deviceId );
    
    return deviceProp.maxThreadsPerMultiProcessor;
}



//arrayAdd()
__global__ void arrayAddKernel( double *deviceA, int alpha, int elementNum ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        deviceA[i] = deviceA[i] + alpha;
    }
}

void arrayAdd( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum, double alpha ){
    int sizeA = elementNum * sizeof(double);
    
    //
    double * deviceA;
    cudaMalloc( (void **) &deviceA, sizeA );

    //
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );
    
    //
    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;
    
    arrayAddKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, alpha, elementNum );
    
    //
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );
    
    
    cudaFree(deviceA);
    
}


//hadamardProduct()
__global__ void hadamardProductKernel( double * deviceA, double * deviceB, int elementNum ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if( i < elementNum ){
        deviceA[i] = deviceA[i] * deviceB[i];
    }
}

void hadamardProduct( deviceContextStruct * deviceContextStructP, double * hostAP, double * hostBP, int elementNum ){
    int sizeA = elementNum * sizeof(double);
    
    //
    double * deviceA, * deviceB;
    cudaMalloc( (void **) &deviceA, sizeA );
    cudaMalloc( (void **) &deviceB, sizeA );

    //
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );
    cudaMemcpy( deviceB, hostBP, sizeA, cudaMemcpyHostToDevice );
    
    //
    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;
    
    hadamardProductKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, deviceB, elementNum );
    
    //
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );
    
    
    cudaFree(deviceA);
    
}






/*
 * Matrix is a PHP extension. It can do parallel computing base on CUDA.
 *
 * GitHub: https://github.com/BourneSuper/matrix
 *
 * Author: Bourne Wong <cb44606@gmail.com>
 *
 * */
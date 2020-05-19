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
__global__ void arrayAddKernel( double *deviceA, double alpha, int elementNum ){
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


//subtractArray()
__global__ void subtractArrayKernel(  double alpha, double *deviceA, int elementNum ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        deviceA[i] = alpha - deviceA[i];
    }
}

void subtractArray( deviceContextStruct * deviceContextStructP,  double alpha, double * hostAP, int elementNum ){
    int sizeA = elementNum * sizeof(double);
    
    //
    double * deviceA;
    cudaMalloc( (void **) &deviceA, sizeA );

    //
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );
    
    //
    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;
    
    subtractArrayKernel<<< blocksPerGrid, threadsPerBlock >>>( alpha, deviceA, elementNum );
    
    //
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );
    
    
    cudaFree(deviceA);
    
}


//arrayMultiply()
__global__ void arrayMultiplyKernel( double *deviceA, double alpha, int elementNum ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        deviceA[i] = deviceA[i] * alpha;
    }
}

void arrayMultiply( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum, double alpha ){
    int sizeA = elementNum * sizeof(double);
    
    //
    double * deviceA;
    cudaMalloc( (void **) &deviceA, sizeA );

    //
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );
    
    //
    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;
    
    arrayMultiplyKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, alpha, elementNum );
    
    //
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );
    
    
    cudaFree(deviceA);
    
}


//divideArray()
__global__ void divideArrayKernel(  double alpha, double *deviceA, int elementNum ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        deviceA[i] = alpha / deviceA[i];
    }
}

void divideArray( deviceContextStruct * deviceContextStructP,  double alpha, double * hostAP, int elementNum ){
    int sizeA = elementNum * sizeof(double);
    
    //
    double * deviceA;
    cudaMalloc( (void **) &deviceA, sizeA );

    //
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );
    
    //
    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;
    
    divideArrayKernel<<< blocksPerGrid, threadsPerBlock >>>( alpha, deviceA, elementNum );
    
    //
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );
    
    
    cudaFree(deviceA);
    
}


//arrayPower()
__global__ void arrayPowerKernel( double *deviceA, double alpha, int elementNum ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        deviceA[i] =  pow( deviceA[i], alpha );
    }
}

void arrayPower( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum, double alpha ){
    int sizeA = elementNum * sizeof(double);
    
    //
    double * deviceA;
    cudaMalloc( (void **) &deviceA, sizeA );

    //
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );
    
    //
    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;
    
    arrayPowerKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, alpha, elementNum );
    
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



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


//arraySquareRoot()
__global__ void arraySquareRootKernel( double *deviceA, int elementNum ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        deviceA[i] =  sqrt( deviceA[i] );
    }
}

void arraySquareRoot( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum ){
    int sizeA = elementNum * sizeof(double);
    
    //
    double * deviceA;
    cudaMalloc( (void **) &deviceA, sizeA );

    //
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );
    
    //
    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;
    
    arraySquareRootKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, elementNum );
    
    //
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );
    
    
    cudaFree(deviceA);
    
}


//arrayCubeRoot()
__global__ void arrayCubeRootKernel( double *deviceA, int elementNum ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        deviceA[i] =  cbrt( deviceA[i] );
    }
}

void arrayCubeRoot( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum ){
    int sizeA = elementNum * sizeof(double);
    
    //
    double * deviceA;
    cudaMalloc( (void **) &deviceA, sizeA );

    //
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );
    
    //
    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;
    
    arrayCubeRootKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, elementNum );
    
    //
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );
    
    
    cudaFree(deviceA);
    
}


//logEArray()
__global__ void logEArrayKernel( double *deviceA, int elementNum ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        deviceA[i] =  log( deviceA[i] );
    }
}

void logEArray( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum ){
    int sizeA = elementNum * sizeof(double);
    
    //
    double * deviceA;
    cudaMalloc( (void **) &deviceA, sizeA );

    //
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );
    
    //
    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;
    
    logEArrayKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, elementNum );
    
    //
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );
    
    
    cudaFree(deviceA);
    
}


//log2Array()
__global__ void log2ArrayKernel( double *deviceA, int elementNum ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        deviceA[i] =  log2( deviceA[i] );
    }
}

void log2Array( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum ){
    int sizeA = elementNum * sizeof(double);
    
    //
    double * deviceA;
    cudaMalloc( (void **) &deviceA, sizeA );

    //
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );
    
    //
    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;
    
    log2ArrayKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, elementNum );
    
    //
    cudaMemcpy( hostAP, deviceA, sizeA, cudaMemcpyDeviceToHost );
    
    
    cudaFree(deviceA);
    
}


//log10Array()
__global__ void log10ArrayKernel( double *deviceA, int elementNum ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum) {
        deviceA[i] =  log10( deviceA[i] );
    }
}

void log10Array( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum ){
    int sizeA = elementNum * sizeof(double);
    
    //
    double * deviceA;
    cudaMalloc( (void **) &deviceA, sizeA );

    //
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );
    
    //
    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;
    
    log10ArrayKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, elementNum );
    
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


//transpose()
__global__ void transposeKernel( double *deviceA, int elementNum, int heightA, int widthA, double *deviceB ){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if( i < elementNum ) {
        int bI = i / heightA;
        int bJ = i % heightA;
        deviceB[i] =  deviceA[ bJ * widthA + bI ];
    }
}

void transpose( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum, int heightA, int wigthA ){
    int sizeA = elementNum * sizeof(double);
    
    //
    double * deviceA, * deviceB;
    cudaMalloc( (void **) &deviceA, sizeA );
    cudaMalloc( (void **) &deviceB, sizeA );

    //
    cudaMemcpy( deviceA, hostAP, sizeA, cudaMemcpyHostToDevice );
    
    //
    int threadsPerBlock = getMaxThreadsPerMultiProcessor( deviceContextStructP );
    int blocksPerGrid = ( elementNum + threadsPerBlock - 1 ) / threadsPerBlock;
    
    transposeKernel<<< blocksPerGrid, threadsPerBlock >>>( deviceA, elementNum, heightA, wigthA, deviceB );

    //
    cudaMemcpy( hostAP, deviceB, sizeA, cudaMemcpyDeviceToHost );
    
    
    cudaFree(deviceA);
    cudaFree(deviceB);
    
}






/*
 * Matrix is a PHP extension. It can do parallel computing base on CUDA.
 *
 * GitHub: https://github.com/BourneSuper/matrix
 *
 * Author: Bourne Wong <cb44606@gmail.com>
 *
 * */



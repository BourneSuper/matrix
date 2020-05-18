/*
 * Matrix is a PHP extension. It can do parallel computing base on CUDA.
 *
 * GitHub: https://github.com/BourneSuper/matrix
 *
 * Author: Bourne Wong <cb44606@gmail.com>
 *
 * */

typedef struct {
    int deviceId;
} deviceContextStruct;


void arrayAdd( deviceContextStruct * deviceContextStructP, double * hostAP, int elementNum, double alpha );

void subtractArray( deviceContextStruct * deviceContextStructP, double alpha, double * hostAP, int elementNum );

void hadamardProduct( deviceContextStruct * deviceContextStructP, double * hostAP, double * hostBP, int elementNum );





















/*
 * Matrix is a PHP extension. It can do parallel computing base on CUDA.
 *
 * GitHub: https://github.com/BourneSuper/matrix
 *
 * Author: Bourne Wong <cb44606@gmail.com>
 *
 * */

/*
 * Matrix is a PHP extension. It can do parallel computing base on CUDA.
 *
 * GitHub: https://github.com/BourneSuper/matrix
 *
 * Author: Bourne Wong <cb44606@gmail.com>
 *
 * */

#include <stdio.h>

#include "dev_util_c.h"


char * duc_getErrorMsg( int code ){
    char * errorCodeMap[3000];
    errorCodeMap[1000] = (char *)"CUDA error. Code %d (File:%s Line: %d)\n \n";
    errorCodeMap[1001] = (char *)"There are no available device(s) that support CUDA \n";


    errorCodeMap[2000] = (char *)"the width of martrixA( %d, %d ) can not match with the height of matrixB( %d, %d )\n";
    errorCodeMap[2001] = (char *)"matrixArrA must be two dimension array\n";
    errorCodeMap[2002] = (char *)"matrixArrB must be two dimension array\n";
    errorCodeMap[2003] = (char *)"matrixArrC must be two dimension array\n";
    errorCodeMap[2004] = (char *)"the height of martrixA( %d, %d ) can not match with the height of matrixC( %d, %d )\n";
    errorCodeMap[2005] = (char *)"the width of martrixB( %d, %d ) can not match with the width of widthC( %d, %d )\n";
    errorCodeMap[2006] = (char *)"the width of martrixB( %d, %d ) can not match with the width of widthC( %d, %d )\n";
    errorCodeMap[2007] = (char *)"elementNumX must greater or equal than ( elementNumX < 1 + ( heightA - 1 ) * fabs( (int)strideX ) ) \n";
    errorCodeMap[2008] = (char *)"%s must be two dimension array\n";

    return errorCodeMap[code];
}



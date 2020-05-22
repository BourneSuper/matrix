/*
 * Matrix is a PHP extension. It can do parallel computing base on CUDA.
 *
 * GitHub: https://github.com/BourneSuper/matrix
 *
 * Author: Bourne Wong <cb44606@gmail.com>
 *
 * */

#include <zend_exceptions.h>
#include "dev_util_c.h"

#ifndef DEV_UTIL_P_H
# define DEV_UTIL_P_H

static void ccResult( int result, const char *const file, int const line){
    if( result ){
        zend_throw_exception_ex( NULL, result, duc_getErrorMsg(1000), static_cast<unsigned int>(result), file, line );
    }
}
# define checkCudaResult(result) ccResult((result), __FILE__, __LINE__)

#endif

void dup_HashTableTo1DArr( HashTable * hashTableP, double * arrP );
void dup_HashTableTo1DArrS( HashTable * hashTableP, float * arrP );
void dup_HashTableTo1DArrOne( HashTable * hashTableP, double * arrP );
void dup_HashTableTo1DArrOneS( HashTable * hashTableP, float * arrP );
void dup_hashTableTo1DZval( HashTable * hashTableP, zval oneDimesionzval, int * shapeInfo, int * shapeInfoIndex );
void dup_oneDimesnPointerArrReshapeToZval( double * arrP, zval reshapedZval, int * shapeInfo, int * shapeInfoIndex, int * previousCount );
void dup_oneDimensionZavlToPointerArr( zval * oneDimensionZavl, double * arrP );

/*
 * Matrix is a PHP extension. It can do parallel computing base on CUDA.
 *
 * GitHub: https://github.com/BourneSuper/matrix
 *
 * Author: Bourne Wong <cb44606@gmail.com>
 *
 * */





/* bs_matrix extension for PHP */

#include <stdio.h>

// Utilities and system includes
#include <assert.h>
//#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
//#include <helper_functions.h>
//#include <helper_cuda.h>


#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include "php.h"
#include <zend_exceptions.h>
#include "ext/standard/info.h"

#include <dev_util_c.h>
#include "php_bs_matrix.h"
#include "php_bs_util.h"
#include "php_bs_math.h"

/* For compatibility with older PHP versions */
#ifndef ZEND_PARSE_PARAMETERS_NONE
#define ZEND_PARSE_PARAMETERS_NONE() \
	ZEND_PARSE_PARAMETERS_START(0, 0) \
	ZEND_PARSE_PARAMETERS_END()
#endif


//------------------------------------ class start -------------------------------------
typedef struct {
    int num;
} numStruct;

typedef struct {
    cublasHandle_t handle;
} cublasHandleStruct;


zend_class_entry * BLAS_ce;
extern zend_class_entry * Util_ce;
extern zend_function_entry Util_functions[];
extern zend_class_entry * Math_ce;
extern zend_function_entry Math_functions[];

static int handleResourceNum;




/**
 *
 */
void dev_printMatrix(double * C, unsigned int height, unsigned int width) {
    php_printf("---------------------------\n");
    for (unsigned int i = 0; i < height; ++i) {
        for (unsigned int j = 0; j < width; ++j) {

            php_printf("%f  ", C[i * width + j]);
        }
        php_printf("\n");
    }
}

/**
 *
 */
void util_HashTableTo1DArr( HashTable * hashTableP, double * arrP  ){
    int count = 0;
    zend_long hash;
    zend_string *key;
    zval *zvalue;
    zend_long h;
    zend_string *k;
    zval *zv;
    ZEND_HASH_FOREACH_KEY_VAL( hashTableP, hash, key, zvalue ){

        ZEND_HASH_FOREACH_KEY_VAL( Z_ARRVAL_P(zvalue), h, k, zv ){

            arrP[ count ] = zval_get_double_func(zv);

            count++;
        }ZEND_HASH_FOREACH_END();
    } ZEND_HASH_FOREACH_END();

}

/**
 *
 */
void util_HashTableTo1DArrS( HashTable * hashTableP, float * arrP  ){
    int count = 0;
    zend_long hash;
    zend_string *key;
    zval *zvalue;
    zend_long h;
    zend_string *k;
    zval *zv;
    ZEND_HASH_FOREACH_KEY_VAL( hashTableP, hash, key, zvalue ){

        ZEND_HASH_FOREACH_KEY_VAL( Z_ARRVAL_P(zvalue), h, k, zv ){

            arrP[ count ] = (float)zval_get_double_func(zv);

            count++;
        }ZEND_HASH_FOREACH_END();
    } ZEND_HASH_FOREACH_END();

}

/**
 *
 */
void util_HashTableTo1DArrOne( HashTable * hashTableP, double * arrP  ){
    int count = 0;
    zend_long hash;
    zend_string *key;
    zval *zvalue;
    ZEND_HASH_FOREACH_KEY_VAL( hashTableP, hash, key, zvalue ){
        arrP[ count ] = zval_get_double_func(zvalue);

        count++;
    } ZEND_HASH_FOREACH_END();

}

/**
 *
 */
void util_HashTableTo1DArrOneS( HashTable * hashTableP, float * arrP  ){
    int count = 0;
    zend_long hash;
    zend_string *key;
    zval *zvalue;
    ZEND_HASH_FOREACH_KEY_VAL( hashTableP, hash, key, zvalue ){
        arrP[ count ] = (float)zval_get_double_func(zvalue);

        count++;
    } ZEND_HASH_FOREACH_END();

}




//----------------------------



/**
 *
 */
static void handleResourceDescontructor(zend_resource *rsrc){

    cublasHandleStruct * pointer = (cublasHandleStruct *)rsrc->ptr;
    if (pointer) {
        cublasDestroy(pointer->handle);
        efree(pointer);
        rsrc->ptr = NULL;
    }

}



//__construct
ZEND_BEGIN_ARG_INFO_EX( BLAS_construct_ArgInfo, 0, 0, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, __construct){
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        zend_throw_exception_ex(NULL, 1000, duc_getErrorMsg(1000), NULL );
    }

    //-----------------

    //
    zend_resource * cublasHandleResourceP;

    cublasHandleStruct * cublasHandleStructP = ( cublasHandleStruct * )ecalloc( 1, sizeof(cublasHandleStruct) );
    cublasHandle_t handle;

    checkCudaResult( cublasCreate(&handle) );
    cublasHandleStructP->handle = handle;

    cublasHandleResourceP  = zend_register_resource( cublasHandleStructP, handleResourceNum );

    //
    zval cudaHandle;
    ZVAL_RES( &cudaHandle, cublasHandleResourceP );
    zend_update_property( BLAS_ce, getThis( ), "cublasHandle", sizeof("cublasHandle") - 1, &cudaHandle );


}



//handle setter
ZEND_BEGIN_ARG_INFO_EX( BLAS_setHandle_ArgInfo, 0, 0, 1)
    ZEND_ARG_INFO( 0, cublasHandleP )
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, setHandle){
    zval * cublasHandleP = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_RESOURCE(cublasHandleP)
    ZEND_PARSE_PARAMETERS_END();

    zend_update_property( BLAS_ce, getThis( ), "cublasHandle", sizeof("cublasHandle") - 1, cublasHandleP );

}


//handle getter
ZEND_BEGIN_ARG_INFO_EX( BLAS_getHandle_ArgInfo, 0, 0, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, getHandle){
//    php_printf("getHandle()\n");

    zval *obj = getThis();
    zval *tempZVal;

    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct *temp = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);


    RETURN_ZVAL( cublasHandleP, 1, 0 );

}



//multiply()
ZEND_BEGIN_ARG_INFO_EX( BLAS_multiply_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO( 1, matrixAP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, matrixBP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, matrixCP, 1 )
    ZEND_ARG_INFO( 0, alpha )
    ZEND_ARG_INFO( 0, beta )
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, multiply){
    zval * matrixAP = NULL;
    zval * matrixBP = NULL;
    zval * matrixCP = NULL;
    double alpha = 1.0;
    double beta = 0.0;

    ZEND_PARSE_PARAMETERS_START(2, 5)
        Z_PARAM_ARRAY_EX( matrixAP, 0, 1 )
        Z_PARAM_ARRAY_EX( matrixBP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_EX( matrixCP, 0, 1 )
        Z_PARAM_DOUBLE(alpha)
        Z_PARAM_DOUBLE(beta)
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(matrixAP);
    if( Z_TYPE( (hashTableAP->arData)->val ) != IS_ARRAY ){
        zend_throw_exception_ex(NULL, 2001, duc_getErrorMsg(2001), NULL );
    }
    int heightA = zend_hash_num_elements(hashTableAP);
    int widthA = zend_hash_num_elements( Z_ARRVAL( (hashTableAP->arData)->val ) );

    HashTable * hashTableBP = Z_ARRVAL_P(matrixBP);
    if( Z_TYPE( (hashTableBP->arData)->val ) != IS_ARRAY ){
        zend_throw_exception_ex(NULL, 2002, duc_getErrorMsg(2002), NULL );
    }
    int heightB = zend_hash_num_elements(hashTableBP);
    int widthB = zend_hash_num_elements( Z_ARRVAL( (hashTableBP->arData)->val ) );

    HashTable * hashTableCP = NULL;
    int heightC = 0;
    int widthC = 0;
    if( matrixCP != NULL ){
        hashTableCP = Z_ARRVAL_P(matrixCP);
        if( Z_TYPE( (hashTableCP->arData)->val ) != IS_ARRAY ){
            zend_throw_exception_ex(NULL, 2003, duc_getErrorMsg(2003), NULL );
        }
        heightC = zend_hash_num_elements(hashTableCP);
        widthC = zend_hash_num_elements( Z_ARRVAL( (hashTableCP->arData)->val ) );

        if( heightC != heightA ){
            zend_throw_exception_ex(NULL, 2004, duc_getErrorMsg(2004), heightA, widthA, heightC, widthC );
        }

        if( widthC != widthB ){
            zend_throw_exception_ex(NULL, 2006, duc_getErrorMsg(2006), heightB, widthB, heightC, widthC  );
        }

    }

    if( widthA !=  heightB ){
        zend_throw_exception_ex(NULL, 2000, duc_getErrorMsg(2000), heightA, widthA, heightB, widthB );
    }

    double * hostAP = ( double * )malloc( heightA * widthA * sizeof(double) );
    double * hostBP = ( double * )malloc( heightB * widthB * sizeof(double) );
    double * hostCP = ( double * )calloc( heightA * widthB, sizeof(double) );

    util_HashTableTo1DArr( hashTableAP, hostAP );
    util_HashTableTo1DArr( hashTableBP, hostBP );
    if( hashTableCP != NULL ){
        util_HashTableTo1DArr( hashTableCP, hostCP );
    }

//    dev_printMatrix( hostAP, heightA, widthA );
//    dev_printMatrix( hostBP, heightB, widthB );

    //
    double * deviceAP, * deviceBP, * deviceCP;
    cudaMalloc( (void**)&deviceAP, heightA * widthA * sizeof(double) );
    cudaMalloc( (void**)&deviceBP, heightB * widthB * sizeof(double) );
    cudaMalloc( (void**)&deviceCP, heightA * widthB * sizeof(double) );

    //
    cudaMemcpy( deviceAP, hostAP, heightA * widthA * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( deviceBP, hostBP, heightB * widthB * sizeof(double), cudaMemcpyHostToDevice );
    if( hashTableCP != NULL ){
        cudaMemcpy( deviceCP, hostCP, heightC * widthC * sizeof(double), cudaMemcpyHostToDevice );
    }

    //
    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);



    // CUBLAS library uses column-major storage, but C/C++ use row-major storage.
    // When passing the matrix pointer to CUBLAS, the memory layout alters from
    // row-major to column-major, which is equivalent to an implicit transpose.
    cublasDgemm(cudaHandleStructP->handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            widthB, heightA, widthA,
            &alpha,
            deviceBP, widthB,
            deviceAP, widthA,
            &beta,
            deviceCP, widthB
    );

    cudaEventSynchronize(stop);

    cudaMemcpy( hostCP, deviceCP, heightA * widthB * sizeof(double), cudaMemcpyDeviceToHost );


//    dev_printMatrix(hostCP, heightA, widthB);


    //
    zval returnZval; array_init_size( &returnZval, heightA );

    for( int tempI = 0; tempI < heightA; tempI++ ){

        zval tempZval; array_init_size( &tempZval, widthB );
        for( int tempJ = 0; tempJ < widthB; tempJ++ ){

            add_next_index_double( &tempZval, hostCP[ tempI * widthB + tempJ ]);
        }

        add_next_index_zval( &returnZval, &tempZval );
    }

    RETVAL_ZVAL( &returnZval, 1, 1 );

    //
    free(hostAP);
    free(hostBP);
    free(hostCP);
    cudaFree(deviceAP);
    cudaFree(deviceBP);
    cudaFree(deviceCP);

    return ;

}


//multiplyS()
ZEND_BEGIN_ARG_INFO_EX( BLAS_multiplyS_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO( 1, matrixAP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, matrixBP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, matrixCP, 1 )
    ZEND_ARG_INFO( 0, alpha )
    ZEND_ARG_INFO( 0, beta )
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, multiplyS){
    zval * matrixAP = NULL;
    zval * matrixBP = NULL;
    zval * matrixCP = NULL;
    float alpha; double alphaTemp = 1.0;
    float beta; double betaTemp = 0.0;

    ZEND_PARSE_PARAMETERS_START(2, 5)
        Z_PARAM_ARRAY_EX( matrixAP, 0, 1 )
        Z_PARAM_ARRAY_EX( matrixBP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_EX( matrixCP, 0, 1 )
        Z_PARAM_DOUBLE(alphaTemp)
        Z_PARAM_DOUBLE(betaTemp)
    ZEND_PARSE_PARAMETERS_END();

    alpha = (float)alphaTemp;
    beta = (float)betaTemp;

    HashTable * hashTableAP = Z_ARRVAL_P(matrixAP);
    if( Z_TYPE( (hashTableAP->arData)->val ) != IS_ARRAY ){
        zend_throw_exception_ex(NULL, 2001, duc_getErrorMsg(2001), NULL );
    }
    int heightA = zend_hash_num_elements(hashTableAP);
    int widthA = zend_hash_num_elements( Z_ARRVAL( (hashTableAP->arData)->val ) );

    HashTable * hashTableBP = Z_ARRVAL_P(matrixBP);
    if( Z_TYPE( (hashTableBP->arData)->val ) != IS_ARRAY ){
        zend_throw_exception_ex(NULL, 2002,  duc_getErrorMsg(2002), NULL );
    }
    int heightB = zend_hash_num_elements(hashTableBP);
    int widthB = zend_hash_num_elements( Z_ARRVAL( (hashTableBP->arData)->val ) );

    HashTable * hashTableCP = NULL;
    int heightC = 0;
    int widthC = 0;
    if( matrixCP != NULL ){
        hashTableCP = Z_ARRVAL_P(matrixCP);
        if( Z_TYPE( (hashTableCP->arData)->val ) != IS_ARRAY ){
            zend_throw_exception_ex(NULL, 2003, duc_getErrorMsg(2003), NULL );
        }
        heightC = zend_hash_num_elements(hashTableCP);
        widthC = zend_hash_num_elements( Z_ARRVAL( (hashTableCP->arData)->val ) );

        if( heightC != heightA ){
            zend_throw_exception_ex(NULL, 2004,  duc_getErrorMsg(2004), heightA, widthA, heightC, widthC );
        }

        if( widthC != widthB ){
            zend_throw_exception_ex(NULL, 2005, duc_getErrorMsg(2005), heightB, widthB, heightC, widthC  );
        }

    }

    if( widthA !=  heightB ){
        zend_throw_exception_ex(NULL, 2000, duc_getErrorMsg(2000), heightA, widthA, heightB, widthB );
    }

    float * hostAP = ( float * )malloc( heightA * widthA * sizeof(float) );
    float * hostBP = ( float * )malloc( heightB * widthB * sizeof(float) );
    float * hostCP = ( float * )calloc( heightA * widthB, sizeof(float) );

    util_HashTableTo1DArrS( hashTableAP, hostAP );
    util_HashTableTo1DArrS( hashTableBP, hostBP );
    if( hashTableCP != NULL ){
        util_HashTableTo1DArrS( hashTableCP, hostCP );
    }

//    dev_printMatrix( hostAP, heightA, widthA );
//    dev_printMatrix( hostBP, heightB, widthB );

    //
    float * deviceAP, * deviceBP, * deviceCP;
    cudaMalloc( (void**)&deviceAP, heightA * widthA * sizeof(float) );
    cudaMalloc( (void**)&deviceBP, heightB * widthB * sizeof(float) );
    cudaMalloc( (void**)&deviceCP, heightA * widthB * sizeof(float) );

    //
    cudaMemcpy( deviceAP, hostAP, heightA * widthA * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( deviceBP, hostBP, heightB * widthB * sizeof(float), cudaMemcpyHostToDevice );
    if( hashTableCP != NULL ){
        cudaMemcpy( deviceCP, hostCP, heightC * widthC * sizeof(float), cudaMemcpyHostToDevice );
    }

    //
    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);

    // CUBLAS library uses column-major storage, but C/C++ use row-major storage.
    // When passing the matrix pointer to CUBLAS, the memory layout alters from
    // row-major to column-major, which is equivalent to an implicit transpose.
    cublasSgemm(cudaHandleStructP->handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            widthB, heightA, widthA,
            &alpha,
            deviceBP, widthB,
            deviceAP, widthA,
            &beta,
            deviceCP, widthB
    );

    cudaEventSynchronize(stop);

    cudaMemcpy( hostCP, deviceCP, heightA * widthB * sizeof(float), cudaMemcpyDeviceToHost );


//    dev_printMatrix(hostCP, heightA, widthB);


    //
    zval returnZval; array_init_size( &returnZval, heightA );

    for( int tempI = 0; tempI < heightA; tempI++ ){

        zval tempZval; array_init_size( &tempZval, widthB );
        for( int tempJ = 0; tempJ < widthB; tempJ++ ){

            add_next_index_double( &tempZval, hostCP[ tempI * widthB + tempJ ]);
        }

        add_next_index_zval( &returnZval, &tempZval );
    }

    RETVAL_ZVAL( &returnZval, 1, 1 );

    //
    free(hostAP);
    free(hostBP);
    free(hostCP);
    cudaFree(deviceAP);
    cudaFree(deviceBP);
    cudaFree(deviceCP);

    return ;

}



//dot()
ZEND_BEGIN_ARG_INFO_EX( BLAS_dot_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrAP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrBP, 0 )
    ZEND_ARG_INFO( 0, strideA )
    ZEND_ARG_INFO( 0, strideB )
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, dot){
    zval * oneDimensionArrAP = NULL;
    zval * oneDimensionArrBP = NULL;
    zend_long strideA = 1;
    zend_long strideB = 1;


    ZEND_PARSE_PARAMETERS_START(2, 4)
        Z_PARAM_ARRAY_EX( oneDimensionArrAP, 0, 1 )
        Z_PARAM_ARRAY_EX( oneDimensionArrBP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(strideA)
        Z_PARAM_LONG(strideB)
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(oneDimensionArrAP);
    int elementNumA = zend_hash_num_elements(hashTableAP);

    HashTable * hashTableBP = Z_ARRVAL_P(oneDimensionArrBP);
    int elementNumB = zend_hash_num_elements(hashTableBP);

    //
    double * hostAP = ( double * )malloc( elementNumA * sizeof(double) );
    double * hostBP = ( double * )malloc( elementNumB * sizeof(double) );
    double * hostCP = ( double * )malloc( 1 * sizeof(double) );

    util_HashTableTo1DArrOne( hashTableAP, hostAP );
    util_HashTableTo1DArrOne( hashTableBP, hostBP );

    //
    double * deviceAP, * deviceBP, * deviceCP;
    cudaMalloc( (void**)&deviceAP, elementNumA * sizeof(double) );
    cudaMalloc( (void**)&deviceBP, elementNumB * sizeof(double) );
    cudaMalloc( (void**)&deviceCP, 1 * sizeof(double) );

    //
    cudaMemcpy( deviceAP, hostAP, elementNumA * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( deviceBP, hostBP, elementNumB * sizeof(double), cudaMemcpyHostToDevice );

    //
    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);


//    const double alpha = 1.0;
//    const double beta = 0.0;
    cublasDdot(cudaHandleStructP->handle,
            elementNumA,
            deviceAP, strideA,
            deviceBP, strideB,
            deviceCP
    );

    cudaEventSynchronize(stop);

    cudaMemcpy( hostCP, deviceCP, 1 * sizeof(double), cudaMemcpyDeviceToHost );


    RETVAL_DOUBLE( *hostCP );

    //
    free(hostAP);
    free(hostBP);
    free(hostCP);
    cudaFree(deviceAP);
    cudaFree(deviceBP);
    cudaFree(deviceCP);

    return ;

}



//dotS()
ZEND_BEGIN_ARG_INFO_EX( BLAS_dotS_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrAP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrBP, 0 )
    ZEND_ARG_INFO( 0, strideA )
    ZEND_ARG_INFO( 0, strideB )
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, dotS){
    zval * oneDimensionArrAP = NULL;
    zval * oneDimensionArrBP = NULL;
    zend_long strideA = 1;
    zend_long strideB = 1;


    ZEND_PARSE_PARAMETERS_START(2, 4)
        Z_PARAM_ARRAY_EX( oneDimensionArrAP, 0, 1 )
        Z_PARAM_ARRAY_EX( oneDimensionArrBP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(strideA)
        Z_PARAM_LONG(strideB)
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(oneDimensionArrAP);
    int elementNumA = zend_hash_num_elements(hashTableAP);

    HashTable * hashTableBP = Z_ARRVAL_P(oneDimensionArrBP);
    int elementNumB = zend_hash_num_elements(hashTableBP);

    //
    float * hostAP = ( float * )malloc( elementNumA * sizeof(float) );
    float * hostBP = ( float * )malloc( elementNumB * sizeof(float) );
    float * hostCP = ( float * )malloc( 1 * sizeof(float) );

    util_HashTableTo1DArrOneS( hashTableAP, hostAP );
    util_HashTableTo1DArrOneS( hashTableBP, hostBP );

    //
    float * deviceAP, * deviceBP, * deviceCP;
    cudaMalloc( (void**)&deviceAP, elementNumA * sizeof(float) );
    cudaMalloc( (void**)&deviceBP, elementNumB * sizeof(float) );
    cudaMalloc( (void**)&deviceCP, 1 * sizeof(float) );

    //
    cudaMemcpy( deviceAP, hostAP, elementNumA * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( deviceBP, hostBP, elementNumB * sizeof(float), cudaMemcpyHostToDevice );

    //
    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);


//    const double alpha = 1.0;
//    const double beta = 0.0;
    cublasSdot(cudaHandleStructP->handle,
            elementNumA,
            deviceAP, strideA,
            deviceBP, strideB,
            deviceCP
    );

    cudaEventSynchronize(stop);

    cudaMemcpy( hostCP, deviceCP, 1 * sizeof(float), cudaMemcpyDeviceToHost );

    RETVAL_DOUBLE( *hostCP );

    //
    free(hostAP);
    free(hostBP);
    free(hostCP);
    cudaFree(deviceAP);
    cudaFree(deviceBP);
    cudaFree(deviceCP);

    return ;

}



//scal()
ZEND_BEGIN_ARG_INFO_EX( BLAS_scal_ArgInfo, 0, 0, 2)
    ZEND_ARG_INFO( 0, alpha )
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrAP, 0 )
    ZEND_ARG_INFO( 0, increase )
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, scal){
    double alpha = 1.0;
    zval * oneDimensionArrAP = NULL;
    zend_long increase = 1;


    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_DOUBLE(alpha)
        Z_PARAM_ARRAY_EX( oneDimensionArrAP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(increase)
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(oneDimensionArrAP);
    int elementNumA = zend_hash_num_elements(hashTableAP);

    //
    double * hostAP = ( double * )malloc( elementNumA * sizeof(double) );

    util_HashTableTo1DArrOne( hashTableAP, hostAP );

    //
    double * deviceAP;
    cudaMalloc( (void**)&deviceAP, elementNumA * sizeof(double) );

    //
    cudaMemcpy( deviceAP, hostAP, elementNumA * sizeof(double), cudaMemcpyHostToDevice );

    //
    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);


//    const double alpha = 1.0;
//    const double beta = 0.0;
    cublasDscal(cudaHandleStructP->handle,
            elementNumA,
            &alpha, deviceAP, increase
    );

    cudaEventSynchronize(stop);

    cudaMemcpy( hostAP, deviceAP, elementNumA * sizeof(double), cudaMemcpyDeviceToHost );

    //
    zval returnZval; array_init_size( &returnZval, elementNumA );

    for( int tempI = 0; tempI < elementNumA; tempI++ ){
        add_next_index_double( &returnZval, hostAP[ tempI ] );
    }

    RETVAL_ZVAL( &returnZval, 1, 1 );

    //
    free(hostAP);
    cudaFree(deviceAP);

    return ;

}


//scalS()
ZEND_BEGIN_ARG_INFO_EX( BLAS_scalS_ArgInfo, 0, 0, 2)
    ZEND_ARG_INFO( 0, alpha )
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrAP, 0 )
    ZEND_ARG_INFO( 0, increase )
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, scalS){
    float alpha; double alphaTemp = 1.0;
    zval * oneDimensionArrAP = NULL;
    zend_long increase = 1;


    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_DOUBLE(alphaTemp)
        Z_PARAM_ARRAY_EX( oneDimensionArrAP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(increase)
    ZEND_PARSE_PARAMETERS_END();

    alpha = (float)alphaTemp;

    HashTable * hashTableAP = Z_ARRVAL_P(oneDimensionArrAP);
    int elementNumA = zend_hash_num_elements(hashTableAP);

    //
    float * hostAP = ( float * )malloc( elementNumA * sizeof(float) );

    util_HashTableTo1DArrOneS( hashTableAP, hostAP );

    //
    float * deviceAP;
    cudaMalloc( (float**)&deviceAP, elementNumA * sizeof(float) );

    //
    cudaMemcpy( deviceAP, hostAP, elementNumA * sizeof(float), cudaMemcpyHostToDevice );

    //
    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);


//    const double alpha = 1.0;
//    const double beta = 0.0;
    cublasSscal(cudaHandleStructP->handle,
            elementNumA,
            &alpha, deviceAP, increase
    );

    cudaEventSynchronize(stop);

    cudaMemcpy( hostAP, deviceAP, elementNumA * sizeof(float), cudaMemcpyDeviceToHost );

    //
    zval returnZval; array_init_size( &returnZval, elementNumA );

    for( int tempI = 0; tempI < elementNumA; tempI++ ){
        add_next_index_double( &returnZval, hostAP[ tempI ]);
    }

    RETVAL_ZVAL( &returnZval, 1, 1 );

    //
    free(hostAP);
    cudaFree(deviceAP);

    return ;

}



//amax()
ZEND_BEGIN_ARG_INFO_EX( BLAS_amax_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrAP, 0 )
    ZEND_ARG_INFO( 0, increase )
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, amax){

    zval * oneDimensionArrAP = NULL;
    zend_long increase = 1;
    int result  = 0;

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY_EX( oneDimensionArrAP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(increase)
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(oneDimensionArrAP);
    int elementNumA = zend_hash_num_elements(hashTableAP);

    //
    double * hostAP = ( double * )malloc( elementNumA * sizeof(double) );

    util_HashTableTo1DArrOne( hashTableAP, hostAP );

    //
    double * deviceAP;
    cudaMalloc( (void**)&deviceAP, elementNumA * sizeof(double) );

    //
    cudaMemcpy( deviceAP, hostAP, elementNumA * sizeof(double), cudaMemcpyHostToDevice );

    //
    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);



    checkCudaResult(
            cublasIdamax(cudaHandleStructP->handle,
                    elementNumA,
                    deviceAP, increase, &result
            )
    );

    cudaEventSynchronize(stop);


    //
    result = result - 1;
    RETVAL_LONG( (zend_long)result );

    //
    free(hostAP);
    cudaFree(deviceAP);

    return ;

}

//amaxS()
ZEND_BEGIN_ARG_INFO_EX( BLAS_amaxS_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrAP, 0 )
    ZEND_ARG_INFO( 0, increase )
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, amaxS){

    zval * oneDimensionArrAP = NULL;
    zend_long increase = 1;
    int result  = 0;

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY_EX( oneDimensionArrAP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(increase)
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(oneDimensionArrAP);
    int elementNumA = zend_hash_num_elements(hashTableAP);

    //
    float * hostAP = ( float * )malloc( elementNumA * sizeof(float) );

    util_HashTableTo1DArrOneS( hashTableAP, hostAP );

    //
    float * deviceAP;
    cudaMalloc( (void**)&deviceAP, elementNumA * sizeof(float) );

    //
    cudaMemcpy( deviceAP, hostAP, elementNumA * sizeof(float), cudaMemcpyHostToDevice );

    //
    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);


    checkCudaResult(
            cublasIsamax(cudaHandleStructP->handle,
                    elementNumA,
                    deviceAP, increase, &result
            )
    );

    cudaEventSynchronize(stop);


    //
    result = result - 1;
    RETVAL_LONG( (zend_long)result );

    //
    free(hostAP);
    cudaFree(deviceAP);

    return ;

}

//amin()
ZEND_BEGIN_ARG_INFO_EX( BLAS_amin_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrAP, 0 )
    ZEND_ARG_INFO( 0, increase )
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, amin){

    zval * oneDimensionArrAP = NULL;
    zend_long increase = 1;
    int result  = 0;

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY_EX( oneDimensionArrAP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(increase)
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(oneDimensionArrAP);
    int elementNumA = zend_hash_num_elements(hashTableAP);

    //
    double * hostAP = ( double * )malloc( elementNumA * sizeof(double) );

    util_HashTableTo1DArrOne( hashTableAP, hostAP );

    //
    double * deviceAP;
    cudaMalloc( (void**)&deviceAP, elementNumA * sizeof(double) );

    //
    cudaMemcpy( deviceAP, hostAP, elementNumA * sizeof(double), cudaMemcpyHostToDevice );

    //
    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);



    checkCudaResult(
            cublasIdamin(cudaHandleStructP->handle,
                    elementNumA,
                    deviceAP, increase, &result
            )
    );

    cudaEventSynchronize(stop);


    //
    result = result - 1;
    RETVAL_LONG( (zend_long)result );

    //
    free(hostAP);
    cudaFree(deviceAP);

    return ;

}

//aminS()
ZEND_BEGIN_ARG_INFO_EX( BLAS_aminS_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrAP, 0 )
    ZEND_ARG_INFO( 0, increase )
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, aminS){

    zval * oneDimensionArrAP = NULL;
    zend_long increase = 1;
    int result  = 0;

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY_EX( oneDimensionArrAP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(increase)
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(oneDimensionArrAP);
    int elementNumA = zend_hash_num_elements(hashTableAP);

    //
    float * hostAP = ( float * )malloc( elementNumA * sizeof(float) );

    util_HashTableTo1DArrOneS( hashTableAP, hostAP );

    //
    float * deviceAP;
    cudaMalloc( (void**)&deviceAP, elementNumA * sizeof(float) );

    //
    cudaMemcpy( deviceAP, hostAP, elementNumA * sizeof(float), cudaMemcpyHostToDevice );

    //
    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);


    checkCudaResult(
            cublasIsamin(cudaHandleStructP->handle,
                    elementNumA,
                    deviceAP, increase, &result
            )
    );

    cudaEventSynchronize(stop);


    //
    result = result - 1;
    RETVAL_LONG( (zend_long)result );

    //
    free(hostAP);
    cudaFree(deviceAP);

    return ;

}

//axpy()
ZEND_BEGIN_ARG_INFO_EX( BLAS_axpy_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrAP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrBP, 0 )
    ZEND_ARG_INFO( 0, alpha )
    ZEND_ARG_INFO( 0, strideA )
    ZEND_ARG_INFO( 0, strideB )
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, axpy){

    zval * oneDimensionArrAP = NULL;
    zval * oneDimensionArrBP = NULL;
    double alpha = 1.0;
    zend_long strideA = 1;
    zend_long strideB = 1;

    ZEND_PARSE_PARAMETERS_START(2, 5)
        Z_PARAM_ARRAY_EX( oneDimensionArrAP, 0, 1 )
        Z_PARAM_ARRAY_EX( oneDimensionArrBP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(alpha)
        Z_PARAM_LONG(strideA)
        Z_PARAM_LONG(strideB)
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(oneDimensionArrAP);
    int elementNumA = zend_hash_num_elements(hashTableAP);

    HashTable * hashTableBP = Z_ARRVAL_P(oneDimensionArrBP);
    int elementNumB = zend_hash_num_elements(hashTableBP);

    //
    double * hostAP = ( double * )malloc( elementNumA * sizeof(double) );
    double * hostBP = ( double * )malloc( elementNumB * sizeof(double) );

    util_HashTableTo1DArrOne( hashTableAP, hostAP );
    util_HashTableTo1DArrOne( hashTableBP, hostBP );

    //
    double * deviceAP, * deviceBP;
    cudaMalloc( (void**)&deviceAP, elementNumA * sizeof(double) );
    cudaMalloc( (void**)&deviceBP, elementNumB * sizeof(double) );

    //
    cudaMemcpy( deviceAP, hostAP, elementNumA * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( deviceBP, hostBP, elementNumB * sizeof(double), cudaMemcpyHostToDevice );

    //
    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);



    checkCudaResult(
            cublasDaxpy(cudaHandleStructP->handle,
                    elementNumA,
                    &alpha,
                    deviceAP, strideA,
                    deviceBP, strideB
            )
    );

    cudaEventSynchronize(stop);

    cudaMemcpy( hostBP, deviceBP, elementNumB * sizeof(double), cudaMemcpyDeviceToHost );


    //
    zval returnZval; array_init_size( &returnZval, elementNumB );

    for( int tempI = 0; tempI < elementNumB; tempI++ ){
        add_next_index_double( &returnZval, hostBP[ tempI ] );
    }

    RETVAL_ZVAL( &returnZval, 1, 1 );

    //
    free(hostAP);
    free(hostBP);
    cudaFree(deviceAP);
    cudaFree(deviceBP);

    return ;

}

//axpyS()
ZEND_BEGIN_ARG_INFO_EX( BLAS_axpyS_ArgInfo, 0, 0, 2)
    ZEND_ARG_INFO( 0, alpha )
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrAP, 0 )
    ZEND_ARG_INFO( 0, increase )
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, axpyS){

    zval * oneDimensionArrAP = NULL;
    zval * oneDimensionArrBP = NULL;
    float alpha;double alphaTemp = 1.0;
    zend_long strideA = 1;
    zend_long strideB = 1;

    ZEND_PARSE_PARAMETERS_START(2, 5)
        Z_PARAM_ARRAY_EX( oneDimensionArrAP, 0, 1 )
        Z_PARAM_ARRAY_EX( oneDimensionArrBP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(alphaTemp)
        Z_PARAM_LONG(strideA)
        Z_PARAM_LONG(strideB)
    ZEND_PARSE_PARAMETERS_END();

    alpha = (float)alphaTemp;

    HashTable * hashTableAP = Z_ARRVAL_P(oneDimensionArrAP);
    int elementNumA = zend_hash_num_elements(hashTableAP);

    HashTable * hashTableBP = Z_ARRVAL_P(oneDimensionArrBP);
    int elementNumB = zend_hash_num_elements(hashTableBP);

    //
    float * hostAP = ( float * )malloc( elementNumA * sizeof(float) );
    float * hostBP = ( float * )malloc( elementNumB * sizeof(float) );

    util_HashTableTo1DArrOneS( hashTableAP, hostAP );
    util_HashTableTo1DArrOneS( hashTableBP, hostBP );

    //
    float * deviceAP, * deviceBP;
    cudaMalloc( (void**)&deviceAP, elementNumA * sizeof(float) );
    cudaMalloc( (void**)&deviceBP, elementNumB * sizeof(float) );

    //
    cudaMemcpy( deviceAP, hostAP, elementNumA * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( deviceBP, hostBP, elementNumB * sizeof(float), cudaMemcpyHostToDevice );

    //
    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);



    checkCudaResult(
            cublasSaxpy(cudaHandleStructP->handle,
                    elementNumA,
                    &alpha,
                    deviceAP, strideA,
                    deviceBP, strideB
            )
    );

    cudaEventSynchronize(stop);

    cudaMemcpy( hostBP, deviceBP, elementNumB * sizeof(float), cudaMemcpyDeviceToHost );


    //
    zval returnZval; array_init_size( &returnZval, elementNumB );

    for( int tempI = 0; tempI < elementNumB; tempI++ ){
        add_next_index_double( &returnZval, hostBP[ tempI ] );
    }

    RETVAL_ZVAL( &returnZval, 1, 1 );

    //
    free(hostAP);
    free(hostBP);
    cudaFree(deviceAP);
    cudaFree(deviceBP);

    return ;

}

//gemv()
ZEND_BEGIN_ARG_INFO_EX( BLAS_gemv_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO( 1, matrixAP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrXP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrYP, 0 )
    ZEND_ARG_INFO( 0, alpha )
    ZEND_ARG_INFO( 0, beta )
    ZEND_ARG_INFO( 0, strideX )
    ZEND_ARG_INFO( 0, strideY )
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, gemv){

    zval * matrixAP = NULL;
    zval * oneDimensionArrXP = NULL;
    zval * oneDimensionArrYP = NULL;
    double alpha = 1.0;
    double beta = 1.0;
    zend_long strideX = 1;
    zend_long strideY = 1;

    ZEND_PARSE_PARAMETERS_START(2, 7)
        Z_PARAM_ARRAY_EX( matrixAP, 0, 1 )
        Z_PARAM_ARRAY_EX( oneDimensionArrXP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_EX( oneDimensionArrYP, 0, 1 )
        Z_PARAM_DOUBLE(alpha)
        Z_PARAM_DOUBLE(beta)
        Z_PARAM_LONG(strideX)
        Z_PARAM_LONG(strideY)
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(matrixAP);
    int heightA = zend_hash_num_elements(hashTableAP);

    if( Z_TYPE( (hashTableAP->arData)->val ) != IS_ARRAY ){
        zend_throw_exception_ex(NULL, 2006, duc_getErrorMsg(2006), NULL );
    }

    int widthA = zend_hash_num_elements( Z_ARRVAL( (hashTableAP->arData)->val ) );



    HashTable * hashTableXP = Z_ARRVAL_P(oneDimensionArrXP);
    int elementNumX = zend_hash_num_elements(hashTableXP);

    HashTable * hashTableYP = NULL;
    int elementNumY = 1  + ( widthA - 1 ) * fabs( (int)strideY );//CUBLAS library uses column-major storage...

    if( oneDimensionArrYP != NULL ){
        hashTableYP = Z_ARRVAL_P(oneDimensionArrYP);
        int tempNumY = zend_hash_num_elements(hashTableYP);
        if( tempNumY != 0 ){
            elementNumY = tempNumY;
        }
    }

    if( elementNumX < 1 + ( heightA - 1 ) * fabs( (int)strideX ) ){
        zend_throw_exception_ex( NULL, 2007, duc_getErrorMsg(2007), NULL );
    }


    //
    double * hostAP = ( double * )malloc( heightA * widthA * sizeof(double) );
    double * hostXP = ( double * )malloc( elementNumX * sizeof(double) );
    double * hostYP = ( double * )calloc( elementNumY, sizeof(double) );

    util_HashTableTo1DArr( hashTableAP, hostAP );
    util_HashTableTo1DArrOne( hashTableXP, hostXP );

    if( oneDimensionArrYP != NULL ){
        util_HashTableTo1DArrOne( hashTableYP, hostYP );
    }

    //
    double * deviceAP, * deviceXP, * deviceYP;
    cudaMalloc( (void**)&deviceAP, heightA * widthA * sizeof(double) );
    cudaMalloc( (void**)&deviceXP, elementNumX * sizeof(double) );
    cudaMalloc( (void**)&deviceYP, elementNumY * sizeof(double) );

    //
    cudaMemcpy( deviceAP, hostAP, heightA * widthA * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( deviceXP, hostXP, elementNumX * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( deviceYP, hostYP, elementNumY * sizeof(double), cudaMemcpyHostToDevice );

    //
    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);



    // CUBLAS library uses column-major storage, but C/C++ use row-major storage.
    // When passing the matrix pointer to CUBLAS, the memory layout alters from
    // row-major to column-major, which is equivalent to an implicit transpose.
    // so exchange height and width
    checkCudaResult(
            cublasDgemv(cudaHandleStructP->handle,
                    CUBLAS_OP_N,
                    widthA, heightA,
                    &alpha,
                    deviceAP, widthA,
                    deviceXP, (int)strideX,
                    &beta,
                    deviceYP, (int)strideY
            )
    );

    cudaEventSynchronize(stop);

    cudaMemcpy( hostYP, deviceYP, elementNumY * sizeof(double), cudaMemcpyDeviceToHost );


    //
    zval returnZval; array_init_size( &returnZval, elementNumY );

    for( int tempI = 0; tempI < elementNumY; tempI++ ){
        add_next_index_double( &returnZval, hostYP[ tempI ] );
    }

    RETVAL_ZVAL( &returnZval, 1, 1 );

    //
    free(hostAP);
    free(hostXP);
    free(hostYP);
    cudaFree(deviceAP);
    cudaFree(deviceXP);
    cudaFree(deviceYP);

    return ;

}


//gemvS()
ZEND_BEGIN_ARG_INFO_EX( BLAS_gemvS_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO( 1, matrixAP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrXP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, oneDimensionArrYP, 0 )
    ZEND_ARG_INFO( 0, alpha )
    ZEND_ARG_INFO( 0, beta )
    ZEND_ARG_INFO( 0, strideX )
    ZEND_ARG_INFO( 0, strideY )
ZEND_END_ARG_INFO()

PHP_METHOD(BLAS, gemvS){

    zval * matrixAP = NULL;
    zval * oneDimensionArrXP = NULL;
    zval * oneDimensionArrYP = NULL;
    float alpha;double alphaTemp = 1.0;
    float beta;double betaTemp = 1.0;
    zend_long strideX = 1;
    zend_long strideY = 1;

    ZEND_PARSE_PARAMETERS_START(2, 7)
        Z_PARAM_ARRAY_EX( matrixAP, 0, 1 )
        Z_PARAM_ARRAY_EX( oneDimensionArrXP, 0, 1 )
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_EX( oneDimensionArrYP, 0, 1 )
        Z_PARAM_DOUBLE(alphaTemp)
        Z_PARAM_DOUBLE(betaTemp)
        Z_PARAM_LONG(strideX)
        Z_PARAM_LONG(strideY)
    ZEND_PARSE_PARAMETERS_END();

    alpha = alphaTemp;
    beta = betaTemp;

    HashTable * hashTableAP = Z_ARRVAL_P(matrixAP);
    int heightA = zend_hash_num_elements(hashTableAP);

    if( Z_TYPE( (hashTableAP->arData)->val ) != IS_ARRAY ){
        zend_throw_exception_ex(NULL, 2006, duc_getErrorMsg(2006), NULL );
    }

    int widthA = zend_hash_num_elements( Z_ARRVAL( (hashTableAP->arData)->val ) );



    HashTable * hashTableXP = Z_ARRVAL_P(oneDimensionArrXP);
    int elementNumX = zend_hash_num_elements(hashTableXP);

    HashTable * hashTableYP = NULL;
    int elementNumY = 1  + ( widthA - 1 ) * fabs( (int)strideY );//CUBLAS library uses column-major storage...

    if( oneDimensionArrYP != NULL ){
        hashTableYP = Z_ARRVAL_P(oneDimensionArrYP);
        int tempNumY = zend_hash_num_elements(hashTableYP);
        if( tempNumY != 0 ){
            elementNumY = tempNumY;
        }
    }

    if( elementNumX < 1 + ( heightA - 1 ) * fabs( (int)strideX ) ){
        zend_throw_exception_ex( NULL, 2007, duc_getErrorMsg(2007), NULL );
    }


    //
    float * hostAP = ( float * )malloc( heightA * widthA * sizeof(float) );
    float * hostXP = ( float * )malloc( elementNumX * sizeof(float) );
    float * hostYP = ( float * )calloc( elementNumY, sizeof(float) );

    util_HashTableTo1DArrS( hashTableAP, hostAP );
    util_HashTableTo1DArrOneS( hashTableXP, hostXP );

    if( oneDimensionArrYP != NULL ){
        util_HashTableTo1DArrOneS( hashTableYP, hostYP );
    }

    //
    float * deviceAP, * deviceXP, * deviceYP;
    cudaMalloc( (void**)&deviceAP, heightA * widthA * sizeof(float) );
    cudaMalloc( (void**)&deviceXP, elementNumX * sizeof(float) );
    cudaMalloc( (void**)&deviceYP, elementNumY * sizeof(float) );

    //
    cudaMemcpy( deviceAP, hostAP, heightA * widthA * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( deviceXP, hostXP, elementNumX * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( deviceYP, hostYP, elementNumY * sizeof(float), cudaMemcpyHostToDevice );

    //
    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(BLAS_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);



    // CUBLAS library uses column-major storage, but C/C++ use row-major storage.
    // When passing the matrix pointer to CUBLAS, the memory layout alters from
    // row-major to column-major, which is equivalent to an implicit transpose.
    // so exchange height and width
    checkCudaResult(
            cublasSgemv(cudaHandleStructP->handle,
                    CUBLAS_OP_N,
                    widthA, heightA,
                    &alpha,
                    deviceAP, widthA,
                    deviceXP, (int)strideX,
                    &beta,
                    deviceYP, (int)strideY
            )
    );

    cudaEventSynchronize(stop);

    cudaMemcpy( hostYP, deviceYP, elementNumY * sizeof(float), cudaMemcpyDeviceToHost );


    //
    zval returnZval; array_init_size( &returnZval, elementNumY );

    for( int tempI = 0; tempI < elementNumY; tempI++ ){
        add_next_index_double( &returnZval, hostYP[ tempI ] );
    }

    RETVAL_ZVAL( &returnZval, 1, 1 );

    //
    free(hostAP);
    free(hostXP);
    free(hostYP);
    cudaFree(deviceAP);
    cudaFree(deviceXP);
    cudaFree(deviceYP);

    return ;

}



//
const zend_function_entry BLAS_functions[] = {
    PHP_ME(BLAS, __construct, BLAS_construct_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(BLAS, getHandle, BLAS_getHandle_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(BLAS, setHandle, BLAS_setHandle_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(BLAS, multiply, BLAS_multiply_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(BLAS, multiplyS, BLAS_multiplyS_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(BLAS, dot, BLAS_dot_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(BLAS, dotS, BLAS_dotS_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(BLAS, scal, BLAS_scal_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(BLAS, scalS, BLAS_scalS_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(BLAS, amax, BLAS_amax_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(BLAS, amaxS, BLAS_amaxS_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(BLAS, amin, BLAS_amin_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(BLAS, aminS, BLAS_aminS_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(BLAS, axpy, BLAS_axpy_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(BLAS, axpyS, BLAS_axpyS_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(BLAS, gemv, BLAS_gemv_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(BLAS, gemvS, BLAS_gemvS_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_FE_END
};


//-------------------

//
PHP_MINIT_FUNCTION(bs_matrix){
    //
    zend_class_entry temp_BLAS_ce;

    INIT_NS_CLASS_ENTRY(temp_BLAS_ce, "BS\\matrix", "BLAS", BLAS_functions);
    BLAS_ce = zend_register_internal_class(&temp_BLAS_ce TSRMLS_CC);

    handleResourceNum = zend_register_list_destructors_ex(handleResourceDescontructor, NULL, "handleResourceName", module_number);
    zend_declare_property_null(BLAS_ce, "cublasHandle", sizeof("cublasHandle") - 1, ZEND_ACC_PROTECTED TSRMLS_CC);


    //
    zend_class_entry temp_Util_ce;
    INIT_NS_CLASS_ENTRY(temp_Util_ce, "BS\\matrix", "Util", Util_functions);

    Util_ce = zend_register_internal_class(&temp_Util_ce TSRMLS_CC);

    //
    zend_class_entry temp_Math_ce;
    INIT_NS_CLASS_ENTRY(temp_Math_ce, "BS\\matrix", "Math", Math_functions);

    Math_ce = zend_register_internal_class(&temp_Math_ce TSRMLS_CC);
    zend_declare_property_long( Math_ce, "DEVICE_ID", sizeof("DEVICE_ID") - 1, 0, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC TSRMLS_CC );

    return SUCCESS;
}


//------------------------------------ class end --------------------------------------
//


ZEND_BEGIN_ARG_INFO_EX(arginfo_bs_matrix_test1, 0, 0, 1)
    ZEND_ARG_INFO( 0, arr )

ZEND_END_ARG_INFO()

/* {{{ void bs_matrix_test1()
 */
PHP_FUNCTION(bs_matrix_test1){
    php_printf("The extension %s is loaded and working!\r\n", "bs_matrix");

    zval *arrPointer = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(arrPointer)
    ZEND_PARSE_PARAMETERS_END();


    HashTable *hashTablePointer = Z_ARRVAL_P(arrPointer);
//    zend_hash_get_current_data_ex( hashTablePointer, zend_hash_get_current_pos(hashTablePointer) );
//    HashTable *widthHashTablePointer = Z_ARRVAL_P();


    zend_long hash;
    zend_string *key;
    zval *zvalue;
    zend_long h;
    zend_string *k;
    zval *zv;

    Bucket *p;

    int height = zend_hash_num_elements(hashTablePointer);
    int width = zend_hash_num_elements( Z_ARRVAL( (hashTablePointer->arData)->val ) );

    double * hostAPointer = (double*)malloc( height * width * sizeof(double));

    int count = 0;
    ZEND_HASH_FOREACH_KEY_VAL(hashTablePointer, hash, key, zvalue){

        ZEND_HASH_FOREACH_KEY_VAL(Z_ARRVAL_P(zvalue), h, k, zv){

            hostAPointer[ count ] = zval_get_double_func(zv);

            count++;
        }ZEND_HASH_FOREACH_END();
    } ZEND_HASH_FOREACH_END();


    zval returnZval; array_init_size( &returnZval, height );

    for( int tempI = 0; tempI < height; tempI++ ){

        zval tempZval; array_init_size( &tempZval, width );
        for( int tempJ = 0; tempJ < width; tempJ++ ){

            add_next_index_double( &tempZval, hostAPointer[tempI * width + tempJ ]);
        }

        add_next_index_zval( &returnZval, &tempZval );
    }

    RETURN_ZVAL( &returnZval, 1, 1 );


}
/* }}} */


/* {{{ PHP_RINIT_FUNCTION
 */
PHP_RINIT_FUNCTION(bs_matrix)
{
#if defined(ZTS) && defined(COMPILE_DL_BS_MATRIX)
	ZEND_TSRMLS_CACHE_UPDATE();
#endif

	return SUCCESS;
}
/* }}} */



/* {{{ PHP_MINFO_FUNCTION
 */
PHP_MINFO_FUNCTION(bs_matrix)
{
	php_info_print_table_start();
	php_info_print_table_header(2, "bs_matrix support", "enabled");
	php_info_print_table_end();
}
/* }}} */

/* {{{ arginfo
 */

/* {{{ bs_matrix_functions[]
 */
static const zend_function_entry bs_matrix_functions[] = {
	PHP_FE(bs_matrix_test1,		arginfo_bs_matrix_test1)
//	PHP_FE(bs_matrix_test2,		arginfo_bs_matrix_test2)
	PHP_FE_END
};
/* }}} */

//--------------------------------------------------------------------------------

/* {{{ bs_matrix_module_entry
 */
zend_module_entry bs_matrix_module_entry = {
	STANDARD_MODULE_HEADER,
	"bs_matrix",					/* Extension name */
	bs_matrix_functions,			/* zend_function_entry */
	PHP_MINIT(bs_matrix),							/* PHP_MINIT - Module initialization */
	NULL,							/* PHP_MSHUTDOWN - Module shutdown */
	PHP_RINIT(bs_matrix),			/* PHP_RINIT - Request initialization */
	NULL,							/* PHP_RSHUTDOWN - Request shutdown */
	PHP_MINFO(bs_matrix),			/* PHP_MINFO - Module info */
	PHP_BS_MATRIX_VERSION,		/* Version */
	STANDARD_MODULE_PROPERTIES
};
/* }}} */

#ifdef COMPILE_DL_BS_MATRIX
# ifdef ZTS
ZEND_TSRMLS_CACHE_DEFINE()
# endif
ZEND_GET_MODULE(bs_matrix)
#endif






/*
 * Matrix is a PHP extension. It can do parallel computing base on CUDA.
 *
 * GitHub: https://github.com/BourneSuper/matrix
 *
 * Author: Bourne Wong <cb44606@gmail.com>
 *
 * */

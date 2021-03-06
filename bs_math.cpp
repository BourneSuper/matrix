/*
 * Matrix is a PHP extension. It can do parallel computing base on CUDA.
 *
 * GitHub: https://github.com/BourneSuper/matrix
 *
 * Author: Bourne Wong <cb44606@gmail.com>
 *
 * */



//class BS.Util


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
#include "math.cuh"
#include "php_bs_math.h"

#include "dev_util_c.h"
#include "dev_util_p.h"





zend_class_entry * Math_ce;





deviceContextStruct * getDeviceContext(){
    deviceContextStruct * deviceContextStructP = (deviceContextStruct *) malloc( sizeof(deviceContextStruct) );

    zval * tempZVal;
    tempZVal = zend_read_static_property( Math_ce, "DEVICE_ID", sizeof( "DEVICE_ID" ) - 1, 0 );

    deviceContextStructP->deviceId = Z_LVAL_P(tempZVal);

    return deviceContextStructP;
}

//--------------------------------------

//setDeviceId() setter
ZEND_BEGIN_ARG_INFO_EX( Math_setDeviceId_ArgInfo, 0, 0, 1)
    ZEND_ARG_INFO( 0, deviceId )
ZEND_END_ARG_INFO()

PHP_METHOD( Math, setDeviceId ){
    zend_long deviceId = 0;

    ZEND_PARSE_PARAMETERS_START( 1, 1 )
        Z_PARAM_LONG( deviceId )
    ZEND_PARSE_PARAMETERS_END();

    zend_update_static_property_long( Math_ce, "DEVICE_ID", sizeof( "DEVICE_ID" ) - 1, deviceId );

}


//getDeviceId() getter
ZEND_BEGIN_ARG_INFO_EX( Math_getDeviceId_ArgInfo, 0, 0, 0)
ZEND_END_ARG_INFO()

PHP_METHOD( Math, getDeviceId ){

    zval *tempZVal;

    tempZVal = zend_read_static_property( Math_ce, "DEVICE_ID", sizeof( "DEVICE_ID" ) - 1, 0 );

    RETURN_ZVAL( tempZVal, 1, 0 );

}

//arrayAdd()
ZEND_BEGIN_ARG_INFO_EX( Math_arrayAdd_ArgInfo, 0, 0, 2 )
    ZEND_ARG_ARRAY_INFO( 1, arrAP, 0 )
    ZEND_ARG_INFO( 0, alpha )
ZEND_END_ARG_INFO()

PHP_METHOD( Math, arrayAdd ){

    zval * arrAP = NULL;
    double alpha = 1.0;

    ZEND_PARSE_PARAMETERS_START( 2, 2 )
        Z_PARAM_ARRAY_EX( arrAP, 0, 1 )
        Z_PARAM_DOUBLE(alpha)
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(arrAP);
    zval oneDimensionZval; array_init( &oneDimensionZval );
    int * shapeInfo = ( int * )calloc( 10, sizeof(int) );
    int shapeInfoIndex = 0;

    dup_hashTableTo1DZval( hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex );

    //
    int elementNum = zend_hash_num_elements( Z_ARRVAL(oneDimensionZval) );
    double * hostAP = ( double * )calloc( elementNum, sizeof(double) );
    dup_oneDimensionZavlToPointerArr( &oneDimensionZval, hostAP );
    arrayAdd( getDeviceContext(), hostAP, elementNum, alpha );

    //
    zval reshapedZval;array_init( &reshapedZval );
    shapeInfoIndex = 0;
    int previousCount = 0;
    dup_oneDimesnPointerArrReshapeToZval( hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount );

    //
    RETVAL_ZVAL( &reshapedZval, 1, 1 );

    free(hostAP);

}

//subtractArray()
ZEND_BEGIN_ARG_INFO_EX( Math_subtractArray_ArgInfo, 0, 0, 2 )
    ZEND_ARG_INFO( 0, alpha )
    ZEND_ARG_ARRAY_INFO( 1, arrAP, 0 )
ZEND_END_ARG_INFO()

PHP_METHOD( Math, subtractArray ){

    double alpha = 1.0;
    zval * arrAP = NULL;

    ZEND_PARSE_PARAMETERS_START( 2, 2 )
        Z_PARAM_DOUBLE(alpha)
        Z_PARAM_ARRAY_EX( arrAP, 0, 1 )
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(arrAP);
    zval oneDimensionZval; array_init( &oneDimensionZval );
    int * shapeInfo = ( int * )calloc( 10, sizeof(int) );
    int shapeInfoIndex = 0;

    dup_hashTableTo1DZval( hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex );

    //
    int elementNum = zend_hash_num_elements( Z_ARRVAL(oneDimensionZval) );
    double * hostAP = ( double * )calloc( elementNum, sizeof(double) );
    dup_oneDimensionZavlToPointerArr( &oneDimensionZval, hostAP );
    subtractArray( getDeviceContext(), alpha, hostAP, elementNum );

    //
    zval reshapedZval;array_init( &reshapedZval );
    shapeInfoIndex = 0;
    int previousCount = 0;
    dup_oneDimesnPointerArrReshapeToZval( hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount );

    //
    RETVAL_ZVAL( &reshapedZval, 1, 1 );

    free(hostAP);

}

//arrayMultiply()
ZEND_BEGIN_ARG_INFO_EX( Math_arrayMultiply_ArgInfo, 0, 0, 2 )
    ZEND_ARG_ARRAY_INFO( 1, arrAP, 0 )
    ZEND_ARG_INFO( 0, alpha )
ZEND_END_ARG_INFO()

PHP_METHOD( Math, arrayMultiply ){

    zval * arrAP = NULL;
    double alpha = 1.0;

    ZEND_PARSE_PARAMETERS_START( 2, 2 )
        Z_PARAM_ARRAY_EX( arrAP, 0, 1 )
        Z_PARAM_DOUBLE(alpha)
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(arrAP);
    zval oneDimensionZval; array_init( &oneDimensionZval );
    int * shapeInfo = ( int * )calloc( 10, sizeof(int) );
    int shapeInfoIndex = 0;

    dup_hashTableTo1DZval( hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex );

    //
    int elementNum = zend_hash_num_elements( Z_ARRVAL(oneDimensionZval) );
    double * hostAP = ( double * )calloc( elementNum, sizeof(double) );
    dup_oneDimensionZavlToPointerArr( &oneDimensionZval, hostAP );
    arrayMultiply( getDeviceContext(), hostAP, elementNum, alpha );

    //
    zval reshapedZval;array_init( &reshapedZval );
    shapeInfoIndex = 0;
    int previousCount = 0;
    dup_oneDimesnPointerArrReshapeToZval( hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount );

    //
    RETVAL_ZVAL( &reshapedZval, 1, 1 );

    free(hostAP);

}

//divideArray()
ZEND_BEGIN_ARG_INFO_EX( Math_divideArray_ArgInfo, 0, 0, 2 )
    ZEND_ARG_INFO( 0, alpha )
    ZEND_ARG_ARRAY_INFO( 1, arrAP, 0 )
ZEND_END_ARG_INFO()

PHP_METHOD( Math, divideArray ){

    double alpha = 1.0;
    zval * arrAP = NULL;

    ZEND_PARSE_PARAMETERS_START( 2, 2 )
        Z_PARAM_DOUBLE(alpha)
        Z_PARAM_ARRAY_EX( arrAP, 0, 1 )
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(arrAP);
    zval oneDimensionZval; array_init( &oneDimensionZval );
    int * shapeInfo = ( int * )calloc( 10, sizeof(int) );
    int shapeInfoIndex = 0;

    dup_hashTableTo1DZval( hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex );

    //
    int elementNum = zend_hash_num_elements( Z_ARRVAL(oneDimensionZval) );
    double * hostAP = ( double * )calloc( elementNum, sizeof(double) );
    dup_oneDimensionZavlToPointerArr( &oneDimensionZval, hostAP );
    divideArray( getDeviceContext(), alpha, hostAP, elementNum );

    //
    zval reshapedZval;array_init( &reshapedZval );
    shapeInfoIndex = 0;
    int previousCount = 0;
    dup_oneDimesnPointerArrReshapeToZval( hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount );

    //
    RETVAL_ZVAL( &reshapedZval, 1, 1 );

    free(hostAP);

}

//arrayPower()
ZEND_BEGIN_ARG_INFO_EX( Math_arrayPower_ArgInfo, 0, 0, 2 )
    ZEND_ARG_ARRAY_INFO( 1, arrAP, 0 )
    ZEND_ARG_INFO( 0, alpha )
ZEND_END_ARG_INFO()

PHP_METHOD( Math, arrayPower ){

    zval * arrAP = NULL;
    double alpha = 1.0;

    ZEND_PARSE_PARAMETERS_START( 2, 2 )
        Z_PARAM_ARRAY_EX( arrAP, 0, 1 )
        Z_PARAM_DOUBLE(alpha)
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(arrAP);
    zval oneDimensionZval; array_init( &oneDimensionZval );
    int * shapeInfo = ( int * )calloc( 10, sizeof(int) );
    int shapeInfoIndex = 0;

    dup_hashTableTo1DZval( hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex );

    //
    int elementNum = zend_hash_num_elements( Z_ARRVAL(oneDimensionZval) );
    double * hostAP = ( double * )calloc( elementNum, sizeof(double) );
    dup_oneDimensionZavlToPointerArr( &oneDimensionZval, hostAP );
    arrayPower( getDeviceContext(), hostAP, elementNum, alpha );

    //
    zval reshapedZval;array_init( &reshapedZval );
    shapeInfoIndex = 0;
    int previousCount = 0;
    dup_oneDimesnPointerArrReshapeToZval( hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount );

    //
    RETVAL_ZVAL( &reshapedZval, 1, 1 );

    free(hostAP);

}

//arraySquareRoot()
ZEND_BEGIN_ARG_INFO_EX( Math_arraySquareRoot_ArgInfo, 0, 0, 1 )
    ZEND_ARG_ARRAY_INFO( 1, arrAP, 0 )
ZEND_END_ARG_INFO()

PHP_METHOD( Math, arraySquareRoot ){

    zval * arrAP = NULL;

    ZEND_PARSE_PARAMETERS_START( 1, 1 )
        Z_PARAM_ARRAY_EX( arrAP, 0, 1 )
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(arrAP);
    zval oneDimensionZval; array_init( &oneDimensionZval );
    int * shapeInfo = ( int * )calloc( 10, sizeof(int) );
    int shapeInfoIndex = 0;

    dup_hashTableTo1DZval( hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex );

    //
    int elementNum = zend_hash_num_elements( Z_ARRVAL(oneDimensionZval) );
    double * hostAP = ( double * )calloc( elementNum, sizeof(double) );
    dup_oneDimensionZavlToPointerArr( &oneDimensionZval, hostAP );
    arraySquareRoot( getDeviceContext(), hostAP, elementNum );

    //
    zval reshapedZval;array_init( &reshapedZval );
    shapeInfoIndex = 0;
    int previousCount = 0;
    dup_oneDimesnPointerArrReshapeToZval( hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount );

    //
    RETVAL_ZVAL( &reshapedZval, 1, 1 );

    free(hostAP);

}

//arrayCubeRoot()
ZEND_BEGIN_ARG_INFO_EX( Math_arrayCubeRoot_ArgInfo, 0, 0, 1 )
    ZEND_ARG_ARRAY_INFO( 1, arrAP, 0 )
ZEND_END_ARG_INFO()

PHP_METHOD( Math, arrayCubeRoot ){

    zval * arrAP = NULL;

    ZEND_PARSE_PARAMETERS_START( 1, 1 )
        Z_PARAM_ARRAY_EX( arrAP, 0, 1 )
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(arrAP);
    zval oneDimensionZval; array_init( &oneDimensionZval );
    int * shapeInfo = ( int * )calloc( 10, sizeof(int) );
    int shapeInfoIndex = 0;

    dup_hashTableTo1DZval( hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex );

    //
    int elementNum = zend_hash_num_elements( Z_ARRVAL(oneDimensionZval) );
    double * hostAP = ( double * )calloc( elementNum, sizeof(double) );
    dup_oneDimensionZavlToPointerArr( &oneDimensionZval, hostAP );
    arrayCubeRoot( getDeviceContext(), hostAP, elementNum );

    //
    zval reshapedZval;array_init( &reshapedZval );
    shapeInfoIndex = 0;
    int previousCount = 0;
    dup_oneDimesnPointerArrReshapeToZval( hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount );

    //
    RETVAL_ZVAL( &reshapedZval, 1, 1 );

    free(hostAP);

}

//logEArray()
ZEND_BEGIN_ARG_INFO_EX( Math_logEArray_ArgInfo, 0, 0, 1 )
    ZEND_ARG_ARRAY_INFO( 1, arrAP, 0 )
ZEND_END_ARG_INFO()

PHP_METHOD( Math, logEArray ){

    zval * arrAP = NULL;

    ZEND_PARSE_PARAMETERS_START( 1, 1 )
        Z_PARAM_ARRAY_EX( arrAP, 0, 1 )
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(arrAP);
    zval oneDimensionZval; array_init( &oneDimensionZval );
    int * shapeInfo = ( int * )calloc( 10, sizeof(int) );
    int shapeInfoIndex = 0;

    dup_hashTableTo1DZval( hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex );

    //
    int elementNum = zend_hash_num_elements( Z_ARRVAL(oneDimensionZval) );
    double * hostAP = ( double * )calloc( elementNum, sizeof(double) );
    dup_oneDimensionZavlToPointerArr( &oneDimensionZval, hostAP );
    logEArray( getDeviceContext(), hostAP, elementNum );

    //
    zval reshapedZval;array_init( &reshapedZval );
    shapeInfoIndex = 0;
    int previousCount = 0;
    dup_oneDimesnPointerArrReshapeToZval( hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount );

    //
    RETVAL_ZVAL( &reshapedZval, 1, 1 );

    free(hostAP);

}

//log2Array()
ZEND_BEGIN_ARG_INFO_EX( Math_log2Array_ArgInfo, 0, 0, 1 )
    ZEND_ARG_ARRAY_INFO( 1, arrAP, 0 )
ZEND_END_ARG_INFO()

PHP_METHOD( Math, log2Array ){

    zval * arrAP = NULL;

    ZEND_PARSE_PARAMETERS_START( 1, 1 )
        Z_PARAM_ARRAY_EX( arrAP, 0, 1 )
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(arrAP);
    zval oneDimensionZval; array_init( &oneDimensionZval );
    int * shapeInfo = ( int * )calloc( 10, sizeof(int) );
    int shapeInfoIndex = 0;

    dup_hashTableTo1DZval( hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex );

    //
    int elementNum = zend_hash_num_elements( Z_ARRVAL(oneDimensionZval) );
    double * hostAP = ( double * )calloc( elementNum, sizeof(double) );
    dup_oneDimensionZavlToPointerArr( &oneDimensionZval, hostAP );
    log2Array( getDeviceContext(), hostAP, elementNum );

    //
    zval reshapedZval;array_init( &reshapedZval );
    shapeInfoIndex = 0;
    int previousCount = 0;
    dup_oneDimesnPointerArrReshapeToZval( hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount );

    //
    RETVAL_ZVAL( &reshapedZval, 1, 1 );

    free(hostAP);

}

//log10Array()
ZEND_BEGIN_ARG_INFO_EX( Math_log10Array_ArgInfo, 0, 0, 1 )
    ZEND_ARG_ARRAY_INFO( 1, arrAP, 0 )
ZEND_END_ARG_INFO()

PHP_METHOD( Math, log10Array ){

    zval * arrAP = NULL;

    ZEND_PARSE_PARAMETERS_START( 1, 1 )
        Z_PARAM_ARRAY_EX( arrAP, 0, 1 )
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(arrAP);
    zval oneDimensionZval; array_init( &oneDimensionZval );
    int * shapeInfo = ( int * )calloc( 10, sizeof(int) );
    int shapeInfoIndex = 0;

    dup_hashTableTo1DZval( hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex );

    //
    int elementNum = zend_hash_num_elements( Z_ARRVAL(oneDimensionZval) );
    double * hostAP = ( double * )calloc( elementNum, sizeof(double) );
    dup_oneDimensionZavlToPointerArr( &oneDimensionZval, hostAP );
    log10Array( getDeviceContext(), hostAP, elementNum );

    //
    zval reshapedZval;array_init( &reshapedZval );
    shapeInfoIndex = 0;
    int previousCount = 0;
    dup_oneDimesnPointerArrReshapeToZval( hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount );

    //
    RETVAL_ZVAL( &reshapedZval, 1, 1 );

    free(hostAP);

}


//hadamardProduct()
ZEND_BEGIN_ARG_INFO_EX( Math_hadamardProduct_ArgInfo, 0, 0, 2 )
    ZEND_ARG_ARRAY_INFO( 1, arrAP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, arrBP, 0 )
ZEND_END_ARG_INFO()

PHP_METHOD( Math, hadamardProduct ){
    zval * arrAP = NULL;
    zval * arrBP = NULL;

    ZEND_PARSE_PARAMETERS_START( 2, 2 )
        Z_PARAM_ARRAY_EX( arrAP, 0, 1 )
        Z_PARAM_ARRAY_EX( arrBP, 0, 1 )
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(arrAP);
    zval oneDimensionAZval; array_init( &oneDimensionAZval );
    int * shapeInfoA = ( int * )calloc( 10, sizeof(int) );
    int shapeInfoIndexA = 0;

    dup_hashTableTo1DZval( hashTableAP, oneDimensionAZval, shapeInfoA, &shapeInfoIndexA );

    HashTable * hashTableBP = Z_ARRVAL_P(arrBP);
    zval oneDimensionBZval; array_init( &oneDimensionBZval );
    int * shapeInfoB = ( int * )calloc( 10, sizeof(int) );
    int shapeInfoIndexB = 0;

    dup_hashTableTo1DZval( hashTableBP, oneDimensionBZval, shapeInfoB, &shapeInfoIndexB );

    //
    int elementNum = zend_hash_num_elements( Z_ARRVAL(oneDimensionAZval) );
    double * hostAP = ( double * )calloc( elementNum, sizeof(double) );
    dup_oneDimensionZavlToPointerArr( &oneDimensionAZval, hostAP );

    double * hostBP = ( double * )calloc( elementNum, sizeof(double) );
    dup_oneDimensionZavlToPointerArr( &oneDimensionBZval, hostBP );

    //
    hadamardProduct( getDeviceContext(), hostAP, hostBP, elementNum );

    //
    zval reshapedZval;array_init( &reshapedZval );
    shapeInfoIndexA = 0;
    int previousCount = 0;
    dup_oneDimesnPointerArrReshapeToZval( hostAP, reshapedZval, shapeInfoA, &shapeInfoIndexA, &previousCount );

    //
    RETVAL_ZVAL( &reshapedZval, 1, 1 );

    free(hostAP);
    free(hostBP);

}

//transpose()
ZEND_BEGIN_ARG_INFO_EX( Math_transpose_ArgInfo, 0, 0, 1 )
    ZEND_ARG_ARRAY_INFO( 1, matrixAP, 0 )
ZEND_END_ARG_INFO()

PHP_METHOD( Math, transpose ){

    zval * matrixAP = NULL;

    ZEND_PARSE_PARAMETERS_START( 1, 1 )
        Z_PARAM_ARRAY_EX( matrixAP, 0, 1 )
    ZEND_PARSE_PARAMETERS_END();

    HashTable * hashTableAP = Z_ARRVAL_P(matrixAP);
    zval oneDimensionZval; array_init( &oneDimensionZval );
    int * shapeInfo = ( int * )calloc( 10, sizeof(int) );
    int shapeInfoIndex = 0;

    dup_hashTableTo1DZval( hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex );

    //
    if( shapeInfo[0] < 1 || shapeInfo[1] < 1 || shapeInfo[2] != 0 ){
        zend_throw_exception_ex(NULL, 2008, duc_getErrorMsg(2008), "matrixA" );
    }

    //
    int elementNum = zend_hash_num_elements( Z_ARRVAL(oneDimensionZval) );
    double * hostAP = ( double * )calloc( elementNum, sizeof(double) );
    dup_oneDimensionZavlToPointerArr( &oneDimensionZval, hostAP );
    transpose( getDeviceContext(), hostAP, elementNum, shapeInfo[0], shapeInfo[1] );

    //
    zval reshapedZval;array_init( &reshapedZval );
    shapeInfoIndex = 0;
    int previousCount = 0;
    int temp = shapeInfo[0]; shapeInfo[0] = shapeInfo[1]; shapeInfo[1] = temp;
    dup_oneDimesnPointerArrReshapeToZval( hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount );

    //
    RETVAL_ZVAL( &reshapedZval, 1, 1 );

    free(hostAP);

}




zend_function_entry Math_functions[] = {
    PHP_ME(Math, setDeviceId, Math_setDeviceId_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_ME(Math, getDeviceId, Math_getDeviceId_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_ME(Math, arrayAdd, Math_arrayAdd_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_ME(Math, subtractArray, Math_subtractArray_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_ME(Math, arrayMultiply, Math_arrayMultiply_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_ME(Math, divideArray, Math_divideArray_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_ME(Math, arrayPower, Math_arrayPower_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_ME(Math, arraySquareRoot, Math_arraySquareRoot_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_ME(Math, arrayCubeRoot, Math_arrayCubeRoot_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_ME(Math, logEArray, Math_logEArray_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_ME(Math, log2Array, Math_log2Array_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_ME(Math, log10Array, Math_log10Array_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_ME(Math, hadamardProduct, Math_hadamardProduct_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_ME(Math, transpose, Math_transpose_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_FE_END
};



//PHP_MINIT_FUNCTION(util){
//    zend_class_entry tmp_ce;
//    INIT_NS_CLASS_ENTRY(tmp_ce, "BS", "Util", Util_functions);
//
//    Util_ce = zend_register_internal_class(&tmp_ce TSRMLS_CC);
//
//    return SUCCESS;
//}







/*
 * Matrix is a PHP extension. It can do parallel computing base on CUDA.
 *
 * GitHub: https://github.com/BourneSuper/matrix
 *
 * Author: Bourne Wong <cb44606@gmail.com>
 *
 * */

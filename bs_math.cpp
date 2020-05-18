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





zend_class_entry * Math_ce;



void hashTableTo1DZval( HashTable * hashTableP, zval oneDimesionzval, int * shapeInfo, int * shapeInfoIndex ){
    zend_long hash;
    zend_string *key;
    zval * zvalue;
    int tempCount = 0;

    ZEND_HASH_FOREACH_KEY_VAL( hashTableP, hash, key, zvalue ){
        if( Z_TYPE_P( zvalue ) == IS_ARRAY ){
            ( * shapeInfoIndex )++;
            hashTableTo1DZval( Z_ARRVAL_P(zvalue), oneDimesionzval, shapeInfo, shapeInfoIndex  );
            ( * shapeInfoIndex )--;
        }else{
            add_next_index_double( &oneDimesionzval, zval_get_double_func(zvalue) );
        }
        tempCount++;

    } ZEND_HASH_FOREACH_END();

    shapeInfo[ * shapeInfoIndex ] = tempCount;

}

void oneDimesnPointerArrReshapeToZval( double * arrP, zval reshapedZval, int * shapeInfo, int * shapeInfoIndex, int * previousCount ){
    //
    int shapeInfoCount = 10;
    for( int i = 0; i < 10; i++ ){
        if( shapeInfo[i] == 0 ){
            shapeInfoCount = i;
            break;
        }
    }

    //
    if( shapeInfo[ * shapeInfoIndex ] == 0 ){
        return ;
    }

    if( * shapeInfoIndex == ( shapeInfoCount - 1 ) ){
        for( int i = 0; i < shapeInfo[ * shapeInfoIndex ]; i++ ){
            add_next_index_double( &reshapedZval, arrP[ ( * previousCount + i ) ] );
        }
        ( * previousCount ) += shapeInfo[ * shapeInfoIndex ];
    }else{
        for( int i = 0; i < shapeInfo[ * shapeInfoIndex ]; i++ ){
            zval tempZval;array_init( &tempZval );
            ( * shapeInfoIndex )++;
            oneDimesnPointerArrReshapeToZval( arrP, tempZval, shapeInfo, shapeInfoIndex, previousCount );
            ( * shapeInfoIndex )--;
            add_next_index_zval( &reshapedZval, &tempZval );
        }
    }



}

void oneDimensionZavlToPointerArr( zval * oneDimensionZavl, double * arrP ){
    HashTable * hashTableP = Z_ARRVAL_P(oneDimensionZavl);

    int count = 0;
    zend_long hash;
    zend_string *key;
    zval *zvalue;
    ZEND_HASH_FOREACH_KEY_VAL( hashTableP, hash, key, zvalue ){
        arrP[ count ] = (float)zval_get_double_func(zvalue);

        count++;
    } ZEND_HASH_FOREACH_END();
}

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

    hashTableTo1DZval( hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex );

    //
    int elementNum = zend_hash_num_elements( Z_ARRVAL(oneDimensionZval) );
    double * hostAP = ( double * )calloc( elementNum, sizeof(double) );
    oneDimensionZavlToPointerArr( &oneDimensionZval, hostAP );
    arrayAdd( getDeviceContext(), hostAP, elementNum, alpha );

    //
    zval reshapedZval;array_init( &reshapedZval );
    shapeInfoIndex = 0;
    int previousCount = 0;
    oneDimesnPointerArrReshapeToZval( hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount );

    //
    RETVAL_ZVAL( &reshapedZval, 1, 1 );

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

    hashTableTo1DZval( hashTableAP, oneDimensionZval, shapeInfo, &shapeInfoIndex );

    //
    int elementNum = zend_hash_num_elements( Z_ARRVAL(oneDimensionZval) );
    double * hostAP = ( double * )calloc( elementNum, sizeof(double) );
    oneDimensionZavlToPointerArr( &oneDimensionZval, hostAP );
    subtractArray( getDeviceContext(), alpha, hostAP, elementNum );

    //
    zval reshapedZval;array_init( &reshapedZval );
    shapeInfoIndex = 0;
    int previousCount = 0;
    oneDimesnPointerArrReshapeToZval( hostAP, reshapedZval, shapeInfo, &shapeInfoIndex, &previousCount );

    //
    RETVAL_ZVAL( &reshapedZval, 1, 1 );

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

    hashTableTo1DZval( hashTableAP, oneDimensionAZval, shapeInfoA, &shapeInfoIndexA );

    HashTable * hashTableBP = Z_ARRVAL_P(arrBP);
    zval oneDimensionBZval; array_init( &oneDimensionBZval );
    int * shapeInfoB = ( int * )calloc( 10, sizeof(int) );
    int shapeInfoIndexB = 0;

    hashTableTo1DZval( hashTableBP, oneDimensionBZval, shapeInfoB, &shapeInfoIndexB );

    //
    int elementNum = zend_hash_num_elements( Z_ARRVAL(oneDimensionAZval) );
    double * hostAP = ( double * )calloc( elementNum, sizeof(double) );
    oneDimensionZavlToPointerArr( &oneDimensionAZval, hostAP );

    double * hostBP = ( double * )calloc( elementNum, sizeof(double) );
    oneDimensionZavlToPointerArr( &oneDimensionBZval, hostBP );

    //
    hadamardProduct( getDeviceContext(), hostAP, hostBP, elementNum );

    //
    zval reshapedZval;array_init( &reshapedZval );
    shapeInfoIndexA = 0;
    int previousCount = 0;
    oneDimesnPointerArrReshapeToZval( hostAP, reshapedZval, shapeInfoA, &shapeInfoIndexA, &previousCount );

    //
    RETVAL_ZVAL( &reshapedZval, 1, 1 );

}




zend_function_entry Math_functions[] = {
    PHP_ME(Math, setDeviceId, Math_setDeviceId_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_ME(Math, getDeviceId, Math_getDeviceId_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_ME(Math, arrayAdd, Math_arrayAdd_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_ME(Math, subtractArray, Math_subtractArray_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_ME(Math, hadamardProduct, Math_hadamardProduct_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
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

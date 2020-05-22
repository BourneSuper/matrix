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
#include "php_bs_util.h"
#include "dev_util_p.h"




zend_class_entry * Util_ce;


//init array by size
ZEND_BEGIN_ARG_INFO_EX(Util_initArrayBySize_ArgInfo, 0, 0, 1 )
    ZEND_ARG_INFO( 0, size )
ZEND_END_ARG_INFO()

PHP_METHOD(Util, initArrayBySize){
    zend_long size;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(size)
    ZEND_PARSE_PARAMETERS_END();

    zval returnZval; array_init_size( &returnZval, size );

    RETURN_ZVAL( &returnZval, 1, 1 );

}


//get device count
ZEND_BEGIN_ARG_INFO_EX(Util_cudaGetDeviceCount_ArgInfo, 0, 0, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(Util, cudaGetDeviceCount){
    int deviceCount = 0;
    checkCudaResult( cudaGetDeviceCount(&deviceCount) );

    RETURN_LONG(deviceCount);
}


//get device name by id
ZEND_BEGIN_ARG_INFO_EX(Util_getDeviceNameById_ArgInfo, 0, 0, 1)
    ZEND_ARG_INFO( 0, deviceId )
ZEND_END_ARG_INFO()

PHP_METHOD(Util, getDeviceNameById){
    zend_long deviceId;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(deviceId)
    ZEND_PARSE_PARAMETERS_END();

    checkCudaResult( cudaSetDevice( (int)deviceId ) );
    cudaDeviceProp deviceProp;
    checkCudaResult( cudaGetDeviceProperties( &deviceProp, (int)deviceId ) );

    RETURN_STRING(deviceProp.name);

}




zend_function_entry Util_functions[] = {
    PHP_ME(Util, initArrayBySize, Util_initArrayBySize_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_ME(Util, cudaGetDeviceCount, Util_cudaGetDeviceCount_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
    PHP_ME(Util, getDeviceNameById, Util_getDeviceNameById_ArgInfo, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC )
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

/* bs_matrix extension for PHP */

#include <stdio.h>

// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>


#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include "php.h"
#include <zend_exceptions.h>
#include "ext/standard/info.h"
#include "php_bs_matrix.h"
#include "php_bs_util.h"

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


zend_class_entry * MatrixTool_ce;
extern zend_class_entry * Util_ce;
extern zend_function_entry Util_functions[];

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


//-------------

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
ZEND_BEGIN_ARG_INFO_EX( MatrixTool_construct_ArgInfo, 0, 0, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(MatrixTool, __construct){
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        zend_throw_exception_ex(NULL, 1000, "There are no available device(s) that support CUDA \n" );
    }

    //-----------------

    //
    zend_resource * cublasHandleResourceP;

    cublasHandleStruct * cublasHandleStructP = ( cublasHandleStruct * )ecalloc( 1, sizeof(cublasHandleStruct) );
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasHandleStructP->handle = handle;

    cublasHandleResourceP  = zend_register_resource( cublasHandleStructP, handleResourceNum );

    //
    zval cudaHandle;
    ZVAL_RES( &cudaHandle, cublasHandleResourceP );
    zend_update_property(MatrixTool_ce, getThis( ), "cublasHandle", sizeof("cublasHandle") - 1, &cudaHandle );


}



//handle setter
ZEND_BEGIN_ARG_INFO_EX( MatrixTool_setHandle_ArgInfo, 0, 0, 1)
    ZEND_ARG_INFO( 0, cublasHandleP )
ZEND_END_ARG_INFO()

PHP_METHOD(MatrixTool, setHandle){
    zval * cublasHandleP = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_RESOURCE(cublasHandleP)
    ZEND_PARSE_PARAMETERS_END();

    zend_update_property(MatrixTool_ce, getThis( ), "cublasHandle", sizeof("cublasHandle") - 1, cublasHandleP );

}


//handle getter
ZEND_BEGIN_ARG_INFO_EX( MatrixTool_getHandle_ArgInfo, 0, 0, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(MatrixTool, getHandle){
//    php_printf("getHandle()\n");

    zval *obj = getThis();
    zval *tempZVal;

    zval *cublasHandleP = zend_read_property(MatrixTool_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct *temp = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);


    RETURN_ZVAL( cublasHandleP, 1, 0 );

}



//multiply
ZEND_BEGIN_ARG_INFO_EX( MatrixTool_multiply_ArgInfo, 0, 0, 2)
    ZEND_ARG_ARRAY_INFO( 1, matrixAP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, matrixBP, 0 )
    ZEND_ARG_ARRAY_INFO( 1, matrixCP, 1 )
    ZEND_ARG_INFO( 0, alpha )
    ZEND_ARG_INFO( 0, beta )
ZEND_END_ARG_INFO()

PHP_METHOD(MatrixTool, multiply){
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
    int heightA = zend_hash_num_elements(hashTableAP);
    int widthA = zend_hash_num_elements( Z_ARRVAL( (hashTableAP->arData)->val ) );

    HashTable * hashTableBP = Z_ARRVAL_P(matrixBP);
    int heightB = zend_hash_num_elements(hashTableBP);
    int widthB = zend_hash_num_elements( Z_ARRVAL( (hashTableBP->arData)->val ) );

    HashTable * hashTableCP = NULL;
    int heightC = 0;
    int widthC = 0;
    if( matrixCP != NULL ){
        hashTableCP = Z_ARRVAL_P(matrixCP);
        heightC = zend_hash_num_elements(hashTableCP);
        widthC = zend_hash_num_elements( Z_ARRVAL( (hashTableCP->arData)->val ) );

        if( heightC != heightA ){
            zend_throw_exception_ex(NULL, 2001, "the height of martrixA( %d, %d ) can not match with the height of matrixC( %d, %d )", heightA, widthA, heightC, widthC );
        }

        if( widthC != widthB ){
            zend_throw_exception_ex(NULL, 2002, "the width of martrixB( %d, %d ) can not match with the width of widthC( %d, %d )", heightB, widthB, heightC, widthC  );
        }

    }

    if( widthA !=  heightB ){
        zend_throw_exception_ex(NULL, 2000, "the width of martrixA( %d, %d ) can not match with the height of matrixB( %d, %d )", heightA, widthA, heightB, widthB );
    }

    double * hostAP = ( double * )malloc( heightA * widthA * sizeof(double) );
    double * hostBP = ( double * )malloc( heightB * widthB * sizeof(double) );
    double * hostCP = ( double * )malloc( heightA * widthB * sizeof(double) );

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
    cudaMemcpy(deviceAP, hostAP, heightA * widthA * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBP, hostBP, heightB * widthB * sizeof(double), cudaMemcpyHostToDevice);
    if( hashTableCP != NULL ){
        cudaMemcpy(deviceCP, hostCP, heightC * widthC * sizeof(double), cudaMemcpyHostToDevice);
    }

    //
    zval *obj = getThis();
    zval *tempZVal;
    zval *cublasHandleP = zend_read_property(MatrixTool_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    cublasHandleStruct * cudaHandleStructP = (cublasHandleStruct *)zend_fetch_resource(Z_RES_P(cublasHandleP), "handleResourceName", handleResourceNum);

    cudaEvent_t stop;
    cudaEventCreate(&stop);



//    const double alpha = 1.0;
//    const double beta = 0.0;
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

    cudaMemcpy(hostCP, deviceCP, heightA * widthB * sizeof(double), cudaMemcpyDeviceToHost);


//    dev_printMatrix(hostCP, heightA, widthB);


    //
    zval returnZval; array_init_size( &returnZval, heightA );

    for( int tempI = 0; tempI < heightA; tempI++ ){

        zval tempZval; array_init_size( &tempZval, widthB );
        for( int tempJ = 0; tempJ < widthB; tempJ++ ){

            add_next_index_double( &tempZval, hostCP[tempI * widthB + tempJ ]);
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


//
const zend_function_entry MatrixTool_functions[] = {
    PHP_ME(MatrixTool, __construct, MatrixTool_construct_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(MatrixTool, multiply, MatrixTool_multiply_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(MatrixTool, getHandle, MatrixTool_getHandle_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(MatrixTool, setHandle, MatrixTool_setHandle_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_FE_END
};


//-------------------

//
PHP_MINIT_FUNCTION(bs_matrix){
    //
    zend_class_entry temp_MatrixTool_ce;

    INIT_NS_CLASS_ENTRY(temp_MatrixTool_ce, "BS", "MatrixTool", MatrixTool_functions);
    MatrixTool_ce = zend_register_internal_class(&temp_MatrixTool_ce TSRMLS_CC);

    handleResourceNum = zend_register_list_destructors_ex(handleResourceDescontructor, NULL, "handleResourceName", module_number);
    zend_declare_property_null(MatrixTool_ce, "cublasHandle", sizeof("cublasHandle") - 1, ZEND_ACC_PROTECTED TSRMLS_CC);


    //
    zend_class_entry temp_Util_ce;
    INIT_NS_CLASS_ENTRY(temp_Util_ce, "BS", "Util", Util_functions);

    Util_ce = zend_register_internal_class(&temp_Util_ce TSRMLS_CC);


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

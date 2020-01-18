/* bs_matrix extension for PHP */

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include "php.h"
#include "ext/standard/info.h"
#include "php_bs_matrix.h"

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


zend_class_entry *MatrixTool_ce;

static int handleResourceNum;



static void handleResourceDescontructor(zend_resource *rsrc){
    php_printf("my_res_dtor() %d \n", rsrc->type);

    numStruct *pointer = (numStruct *)rsrc->ptr;// TO DO
    php_printf("my_res_dtor() %d \n", pointer->num);
    if (pointer) {
        efree(pointer);
        rsrc->ptr = NULL;
    }

    //fclose((FILE *)rsrc->ptr);
//    cublasDestroy((cublasHandle_t *)rsrc->ptr);

}

//cublasHandle_t getCublasHandle( ){
//    zval *obj = getThis();
//    zval *tempZVal;
//    zval *cublasHandleZValP;
//
//
//    cublasHandleZValP = zend_read_property(MatrixTool_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC)
//
//    return (cublasHandle_t *)zend_fetch_resource(Z_RES_P(my_val), "my_res", res_num);
//}



//__construct
ZEND_BEGIN_ARG_INFO_EX( MatrixTool_construct_ArgInfo, 0, 0, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(MatrixTool, __construct){
    php_printf("__construct() \n");

    //
    zend_resource *cudaHandleResourceP;
    numStruct * numStructP = ( numStruct * )ecalloc(1, sizeof(numStruct));
    numStructP->num = 1133;
    cudaHandleResourceP  = zend_register_resource( numStructP, handleResourceNum);

    //
    zval cudaHandle;
    ZVAL_RES( &cudaHandle, cudaHandleResourceP);
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

PHP_METHOD(MatrixTool, getHandle){//not used
//    php_printf("res_num %d \n", res_num);
//    int *rResP = (int *)zend_fetch_resource(Z_RES(my_val), "my_res", res_num);
//
//    php_printf(" %d \n", *rResP);

    php_printf("getHandle()\n");

    zval *obj = getThis();
    zval *tempZVal;
    //zval *cublasHandleZValP;

    //cublasHandleZValP = zend_read_property(MatrixTool_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);

    zval *cublasHandleZValP = zend_read_property(MatrixTool_ce, obj, "cublasHandle", sizeof("cublasHandle") - 1, 1, tempZVal TSRMLS_CC);
    numStruct *temp = (numStruct *)zend_fetch_resource(Z_RES_P(cublasHandleZValP), "my_res", handleResourceNum);
    php_printf("gg %d \n", temp->num);

//    int *temp = (int *)zend_fetch_resource(Z_RES_P(cublasHandleZValP), "my_res", res_num);
//    php_printf("gg %d \n", *temp);


    RETURN_ZVAL( cublasHandleZValP, 1, 0 );

}



//multiply
ZEND_BEGIN_ARG_INFO_EX( MatrixTool_multiply_ArgInfo, 0, 0, 3)
    ZEND_ARG_INFO( 0, cudaHandleP )
    ZEND_ARG_INFO( 1, matrixAP )
    ZEND_ARG_INFO( 1, matrixBP )
ZEND_END_ARG_INFO()

PHP_METHOD(MatrixTool, multiply){
    php_printf("The extension %s is loaded and working!\r\n", "bs_matrix");

    zval * cudaHandleP = NULL;
    zval * matrixAP = NULL;
    zval * matrixBP = NULL;

    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_RESOURCE(cudaHandleP)
        Z_PARAM_ZVAL(matrixAP)
        Z_PARAM_ZVAL(matrixBP)
    ZEND_PARSE_PARAMETERS_END();

    numStruct *temp = (numStruct *)zend_fetch_resource(Z_RES_P(cudaHandleP), "my_res", handleResourceNum);
    php_printf("resource int %d\n", temp->num );

    convert_to_array(matrixAP);
    php_printf("The extension %d \n", Z_TYPE_P( matrixAP ));
    add_next_index_double( matrixAP, 2.222);

    php_printf("fff %d \n", zend_hash_num_elements( Z_ARRVAL_P( matrixAP ) ) );

}


//
const zend_function_entry MatrixTool_functions[] = {
    PHP_ME(MatrixTool, __construct, MatrixTool_construct_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(MatrixTool, multiply, MatrixTool_multiply_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(MatrixTool, getHandle, MatrixTool_getHandle_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_ME(MatrixTool, setHandle, MatrixTool_setHandle_ArgInfo, ZEND_ACC_PUBLIC)
    PHP_FE_END
};


//
PHP_MINIT_FUNCTION(bs_matrix)
{
    zend_class_entry ce;
    INIT_NS_CLASS_ENTRY(ce, "BS", "Matrix", MatrixTool_functions);

    MatrixTool_ce = zend_register_internal_class(&ce TSRMLS_CC);

    /* module_number should be your PHP extension number here */
    handleResourceNum = zend_register_list_destructors_ex(handleResourceDescontructor, NULL, "my_res", module_number);

    zend_declare_property_null(MatrixTool_ce, "cublasHandle", sizeof("cublasHandle") - 1, ZEND_ACC_PROTECTED TSRMLS_CC);

    return SUCCESS;
}


//------------------------------------ class end --------------------------------------
//


ZEND_BEGIN_ARG_INFO_EX(arginfo_bs_matrix_test1, 0, 0, 1)
    ZEND_ARG_INFO( 0, arr )

ZEND_END_ARG_INFO()

/* {{{ void bs_matrix_test1()
 */
PHP_FUNCTION(bs_matrix_test1)
{
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

    int hight = zend_hash_num_elements(hashTablePointer);
    int width = zend_hash_num_elements( Z_ARRVAL( (hashTablePointer->arData)->val ) );

    double * hostAPointer = (double*)malloc( hight * width * sizeof(double));

    int count = 0;
    ZEND_HASH_FOREACH_KEY_VAL(hashTablePointer, hash, key, zvalue){

        ZEND_HASH_FOREACH_KEY_VAL(Z_ARRVAL_P(zvalue), h, k, zv){

            hostAPointer[ count ] = zval_get_double_func(zv);

            count++;
        }ZEND_HASH_FOREACH_END();
    } ZEND_HASH_FOREACH_END();


    zval returnZval; array_init_size( &returnZval, hight );

    for( int tempI = 0; tempI < hight; tempI++ ){

        zval tempZval; array_init_size( &tempZval, width );
        for( int tempJ = 0; tempJ < width; tempJ++ ){

            add_next_index_double( &tempZval, hostAPointer[tempI * width + tempJ ]);
        }

        add_next_index_zval( &returnZval, &tempZval );
    }

    RETURN_ZVAL( &returnZval, 1, 1 );


}
/* }}} */

/* {{{ string bs_matrix_test2( [ string $var ] )
 */
//PHP_FUNCTION(bs_matrix_test2)
//{
//	char *var = "World";
//	size_t var_len = sizeof("World") - 1;
//	zend_string *retval;
//
//	ZEND_PARSE_PARAMETERS_START(0, 1)
//		Z_PARAM_OPTIONAL
//		Z_PARAM_STRING(var, var_len)
//	ZEND_PARSE_PARAMETERS_END();
//
//	retval = strpprintf(0, "Hello %s", var);
//
//	RETURN_STR(retval);
//}
/* }}}*/

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


//ZEND_BEGIN_ARG_INFO(arginfo_bs_matrix_test2, 0)
//	ZEND_ARG_INFO(0, str)
//ZEND_END_ARG_INFO()
/* }}} */




/* {{{ bs_matrix_functions[]
 */
static const zend_function_entry bs_matrix_functions[] = {
	PHP_FE(bs_matrix_test1,		arginfo_bs_matrix_test1)
//	PHP_FE(bs_matrix_test2,		arginfo_bs_matrix_test2)
	PHP_FE_END
};
/* }}} */

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


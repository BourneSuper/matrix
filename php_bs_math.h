/*
 * Matrix is a PHP extension. It can do parallel computing base on CUDA.
 *
 * GitHub: https://github.com/BourneSuper/matrix
 *
 * Author: Bourne Wong <cb44606@gmail.com>
 *
 * */



/* bs_matrix extension for PHP */

#ifndef PHP_BS_MATRIX_H
# define PHP_BS_MATRIX_H

extern zend_module_entry bs_matrix_module_entry;
# define phpext_bs_matrix_ptr &bs_matrix_module_entry

# define PHP_BS_MATRIX_VERSION "0.1.0"

# if defined(ZTS) && defined(COMPILE_DL_BS_MATRIX)
ZEND_TSRMLS_CACHE_EXTERN()
# endif

#endif  /* PHP_BS_MATRIX_H */


#ifndef PHP_BS_UTIL_H
# define PHP_BS_UTIL_H

static void ccResult( int result, const char *const file, int const line){
    if( result ){
        zend_throw_exception_ex( NULL, result, "CUDA error. Code %d (File:%s Line: %d)\n", static_cast<unsigned int>(result), file, line );
    }
}
# define checkCudaResult(result) ccResult((result), __FILE__, __LINE__)

#endif

//PHP_FUNCTION(initArrayBySize);


PHP_METHOD( Math, arrayAdd );
PHP_METHOD( Math, subtractArray );
PHP_METHOD( Math, arrayMultiply );
PHP_METHOD( Math, divideArray );
PHP_METHOD( Math, arrayPower );

PHP_METHOD( Math, hadamardProduct );





/*
 * Matrix is a PHP extension. It can do parallel computing base on CUDA.
 *
 * GitHub: https://github.com/BourneSuper/matrix
 *
 * Author: Bourne Wong <cb44606@gmail.com>
 *
 * */

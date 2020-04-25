/*
 * Matrix is a PHP extension. It can do parallel computing base on CUDA.
 *
 * GitHub: https://github.com/BourneSuper/matrix
 *
 * Author: Bourne Wong <cb44606@gmail.com>
 *
 * */


#ifndef PHP_BS_MATRIX_H
# define PHP_BS_MATRIX_H

extern zend_module_entry bs_matrix_module_entry;
# define phpext_bs_matrix_ptr &bs_matrix_module_entry

# define PHP_BS_MATRIX_VERSION "0.1.0"

# if defined(ZTS) && defined(COMPILE_DL_BS_MATRIX)
ZEND_TSRMLS_CACHE_EXTERN()
# endif

#endif	/* PHP_BS_MATRIX_H */





PHP_METHOD(MatrixTool, __construct);
PHP_METHOD(MatrixTool, setHandle);
PHP_METHOD(MatrixTool, getHandle);
PHP_METHOD(MatrixTool, multiply);
PHP_METHOD(MatrixTool, multiplyS);
PHP_METHOD(MatrixTool, dot);
PHP_METHOD(MatrixTool, dotS);
PHP_METHOD(MatrixTool, scal);
PHP_METHOD(MatrixTool, scalS);
PHP_METHOD(MatrixTool, amax);
PHP_METHOD(MatrixTool, amaxS);
PHP_METHOD(MatrixTool, amin);
PHP_METHOD(MatrixTool, aminS);
PHP_METHOD(MatrixTool, axpy);
PHP_METHOD(MatrixTool, axpyS);
PHP_METHOD(MatrixTool, gemv);
PHP_METHOD(MatrixTool, gemvS);






/*
 * Matrix is a PHP extension. It can do parallel computing base on CUDA.
 *
 * GitHub: https://github.com/BourneSuper/matrix
 *
 * Author: Bourne Wong <cb44606@gmail.com>
 *
 * */

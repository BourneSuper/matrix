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



PHP_FUNCTION(initArrayBySize);


PHP_METHOD(Util, cudaGetDeviceCount);
PHP_METHOD(Util, getDeviceNameById);





/*
 * Matrix is a PHP extension. It can do parallel computing base on CUDA.
 *
 * GitHub: https://github.com/BourneSuper/matrix
 *
 * Author: Bourne Wong <cb44606@gmail.com>
 *
 * */

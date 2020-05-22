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



//PHP_FUNCTION(initArrayBySize);


PHP_METHOD( Math, arrayAdd );
PHP_METHOD( Math, subtractArray );
PHP_METHOD( Math, arrayMultiply );
PHP_METHOD( Math, divideArray );
PHP_METHOD( Math, arrayPower );
PHP_METHOD( Math, arraySquareRoot );
PHP_METHOD( Math, arrayCubeRoot );
PHP_METHOD( Math, logEArray );
PHP_METHOD( Math, log2Array );
PHP_METHOD( Math, log10Array );

PHP_METHOD( Math, hadamardProduct );
PHP_METHOD( Math, transpose );





/*
 * Matrix is a PHP extension. It can do parallel computing base on CUDA.
 *
 * GitHub: https://github.com/BourneSuper/matrix
 *
 * Author: Bourne Wong <cb44606@gmail.com>
 *
 * */

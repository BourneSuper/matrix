[![](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

# Matrix
Matrix is a PHP extension. It can do parallel computing based on CUDA.<br/>
Why should we use GPU to do cumputation ? Because it can run 1000+ times faster than CPU when solve a parallel computation.<br/>
What's more, neural network of AI are full of computation of matrix , so it can be helpful.

## Requirement

- [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus)
- [CUDA Toolkit](https://docs.nvidia.com/cuda/index.html)
- [PHP](https://www.php.net/) version >= 7.0

## Install
1. Generate Makefile
```SHELL
cd matrix/
/usr/local/php/bin/phpize
./configure --with-php-config=/usr/local/php/bin/php-config
```

2. Modify Makefile <br/>
  **sh bs_matrix.mk.sh**

3. make and install <br/>
  **make bs_matrix**
  
4. (optional)copy the file **_bs_matrix_ide_helper.php** to your IDE directory


## Usage

**1. Matrix multiply**
```php
<?php
$matrixA = [ [ 1, 2 ][ 3, 4 ][ 5, 6 ] ]; //A(3,2)
$matrixB = [ [ 7, 8, 9, 0 ][ 1, 2, 3, 4 ] ]; //B(2, 4)
$blas = new BS\matrix\BLAS();
$gpuCalculatedArr = $blas->multiply( $matrixA, $matrixB ); //A(3,2) x B(2, 4)
var_dump( $gpuCalculatedArr );
?>
```
**2. Hadamard product**
```php
<?php
$arrA = [
    [ 1, 2, 3 ],
    [ 4, 5, 6 ],
];

$arrB = [
    [ 2, 2, 2 ],
    [ 2, 2, 2 ],
];

$gpuCalculatedArr = BS\matrix\Math::hadamardProduct( $arrA, $arrB );
var_dump( $gpuCalculatedArr );
?>
```

**3. Get GPU's name**
```php
<?php
$deviceCount = Util::cudaGetDeviceCount();
printf( "CUDA device count: %d \n", $deviceCount );

$deviceId = $deviceCount - 1;
printf( "CUDA device id: %d, name: %s \n", $deviceId, Util::getDeviceNameById($deviceId) );
?>
```
**4. Create and init an array with certain capacity**
```php
<?php
$arr = Util::initArrayBySize( $capacity ); //this can slightly improve performce when access, insert or update an array
?>
```

**Want More?** <br/>
Please see [_bs_matrix_ide_helper.php](https://github.com/BourneSuper/matrix/blob/master/_bs_matrix_ide_helper.php "_bs_matrix_ide_helper.php"). 

## Help with test
install composer first,then
```shell
composer install
sh unit_tests/runTest.sh 
```

## LISENSE
Matrix is licensed under MIT.

And we appeal to [![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE).

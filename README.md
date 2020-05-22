
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
  
4. (optional) Copy the file **_bs_matrix_ide_helper.php** to your IDE directory


## Usage

**1. Matrix multiply**
```php
<?php
function randomArr( $height, $width ){
    $arr = BS\matrix\Util::initArrayBySize($height);
    for($i = 0 ; $i < $height; $i++){
        $tempArr = BS\matrix\Util::initArrayBySize($width);
        for ($j = 0 ; $j < $width; $j++){
            $tempArr[] = rand(1,1000) + rand(1,1000) / 1000 ;
        }
        $arr[] = $tempArr;
    }
    
    return $arr;
}

$matrixA = randomArr( 640, 480 ); //A( 640, 480 )
$matrixB = randomArr( 480, 320 ); //B( 480, 320 )

$blas = new BS\matrix\BLAS();
$gpuCalculatedArr = $blas->multiply( $matrixA, $matrixB ); //A( 640, 480 ) x B( 480, 320 ) , 1000+ times faster than CPU computation

var_dump( $gpuCalculatedArr );//gpuCalculatedArr(640, 320)

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
$deviceCount = BS\matrix\Util::cudaGetDeviceCount();
printf( "CUDA device count: %d \n", $deviceCount );

$deviceId = $deviceCount - 1;
printf( "CUDA device id: %d, name: %s \n", $deviceId, Util::getDeviceNameById($deviceId) );
?>
```
**4. Create and init an array with certain capacity**
```php
<?php
$arr = BS\matrix\Util::initArrayBySize( $capacity ); //this can slightly improve performce when access, insert or update an array
?>
```

**Want More?** <br/>
Please see [_bs_matrix_ide_helper.php](https://github.com/BourneSuper/matrix/blob/master/_bs_matrix_ide_helper.php "_bs_matrix_ide_helper.php"). 

## Help with Test
install composer first,then
```shell
composer install
sh unit_tests/runTest.sh 
```

## LISENSE
Matrix is licensed under MIT.



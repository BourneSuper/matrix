<?php
/*
 * Matrix is a PHP extension. It can do parallel computing base on CUDA.
 *
 * GitHub: https://github.com/BourneSuper/matrix
 *
 * Author: Bourne Wong <cb44606@gmail.com>
 *
 * */

namespace BS;


function sameElementArr($height, $width, $value){
    $arr = Util::initArrayBySize($height);
    for($i = 0 ; $i < $height; $i++){
        $tempArr = Util::initArrayBySize($width);
        for ($j = 0 ; $j < $width; $j++){
            $tempArr[] = $value;
        }
        $arr[] = $tempArr;
    }
    
    return $arr;
}

function randomArr($height, $width){
    $arr = Util::initArrayBySize($height);
    for($i = 0 ; $i < $height; $i++){
        $tempArr = Util::initArrayBySize($width);
        for ($j = 0 ; $j < $width; $j++){
            $tempArr[] = rand(1,1000) + rand(1,1000) / 1000 ;
        }
        $arr[] = $tempArr;
    }
    
    return $arr;
}

function multiply( $a, $b ){
    $c = [];
    for( $h = 0; $h < count($a); $h++ ){
        $tempArr = [];
        for( $s = 0; $s < count($b[0]); $s++ ){
            $tempSum = 0;
            for( $w = 0; $w < count($a[0]); $w++ ){
                $tempSum += $a[$h][$w] * $b[$w][$s];
                
            }
            $tempArr[] = $tempSum;
        }
        $c[] = $tempArr;
    }
    return $c;
}

echo "debug start" . PHP_EOL;
$deviceCount = Util::cudaGetDeviceCount();
printf("CUDA device count: %d \n", $deviceCount );

$deviceId = $deviceCount - 1;
printf("CUDA device id: %d, name: %s \n", $deviceId, Util::getDeviceNameById($deviceId) );

echo "-----------------------" . PHP_EOL;




$a = randomArr( 640, 480 );
$b = randomArr( 480, 320 );
$c = sameElementArr( 640, 320, 1.0 );

// $a = [
//     [1,2,3],
//     [4,5,6]
    
// ];
// $b = [
//     [4,5],
//     [7,8],
//     [9,10]
// ];

$totalLoopNum = 1;

$cpuCalculatedArr = [];
$startTime = microtime(true);
for( $i = 0; $i < $totalLoopNum; $i++ ){
    $cpuCalculatedArr = multiply($a, $b);
}
$cpuConsumeTime = microtime(true) - $startTime;

try {
    $matrixTool = new MatrixTool();
} catch (\Exception $e) {
    echo $e->getMessage();
    die();
}



$handle = $matrixTool->getHandle();
$matrixTool->setHandle($handle);

$gpuCalculatedArr = [];
$startTime = microtime(true);
for( $i = 0; $i < $totalLoopNum; $i++ ){
    $gpuCalculatedArr = $matrixTool->multiply( $a, $b );
//     $gpuCalculatedArr = $matrixTool->multiply( $a, $b, $c, 1.0, 0.0 );
//     $gpuCalculatedArr = $matrixTool->multiplyS( $a, $b );
}
$gpuConsumeTime = microtime(true) - $startTime;

printf("Double type Matrix A(%d,%d) x B(%d,%d) = C comparison, total loop num: %d \n", count($a), count($a[0]), count($b), count($b[0]), $totalLoopNum )  ; 
printf("CPU total consume time: %fs, one loop time: %fs, C(2,3) = %f \n", $cpuConsumeTime, $cpuConsumeTime / $totalLoopNum, $cpuCalculatedArr[2][3]); 
printf("GPU total consume time: %fs, one loop time: %fs, C(2,3) = %f \n", $gpuConsumeTime, $gpuConsumeTime / $totalLoopNum,$gpuCalculatedArr[2][3]); 



// var_dump($a);

// var_dump($matrix);


//
$arrA = [1.1,2,3,4,5];
$arrB = [5,4,3,2,1];
$res = $matrixTool->dot( $arrA, $arrB );
// $res = $matrixTool->dotS( $arrA, $arrB );
var_dump($res);


//
$arrA = [1.1,2,3,4];
$res = $matrixTool->scal( 5.0, $arrA );
// $res = $matrixTool->scalS( 5.0, $arrA );
var_dump($res);



echo "debug finish" . PHP_EOL;
?>

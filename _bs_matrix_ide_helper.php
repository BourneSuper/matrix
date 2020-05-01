<?php
namespace BS\matrix;

class Util{
    
    /**
     * create an empty arrray by setting it's capacity
     * it can slightly imporve performance
     * @param int $size
     */
    public static function initArrayBySize( int $size ){}
    
    /**
     * @throws \Exception
     */
    public static function cudaGetDeviceCount(){}
    
    /**
     * @param int $deviceId
     * @throws \Exception
     */
    public static function getDeviceNameById( $deviceId ){}
    
}


class BLAS{
    
    /**
     * @var resource cublasHandle_t
     */
    protected $cublasHandle;
    
    /**
     * 
     */
    public function __construct(){}
    
    /**
     * @param resource $handle
     */
    public function setHandle( $handle ){}
    
    /**
     * @return resource cublasHandle_t
     */
    public function getHandle(){}
    
    /**
     * This function performs the matrix-matrix multiplication
     * 
     * [[   1,   2,   3 ]     
     *  [ 4.4, 5.5, 6.6 ]]   
     *          *             
     *   [[ 1,   2, 3.3 ]                      
     *    [ 4,   5, 6.6 ]                      
     *    [ 7,   8, 9.9 ]]
     *          =
     *   [[   30,   36,  46.2 ]      
     *    [ 72.6, 89.1, 116.16 ]]                         
     * 
     * $matrixArrC = alpha * ( $matrixArrA * $matrixArrB ) + beta * $matrixArrC
     * 
     * @param array $matrixArrA two dimension array within double type number
     * @param array $matrixArrB two dimension array within double type number
     * @param array $matrixArrC two dimension array within double type number
     * @param double $alpha
     * @param double $beta
     * @return array two dimension array within double type number
     * @throws \Exception
     */
    public function multiply( $matrixArrA, $matrixArrB, $matrixArrC = [][], $alpha = 1.0, $beta = 0.0 ){}
    
    /**
     * This function performs the matrix-matrix multiplication
     * notice precision loss!
     * 
     * 
     * @see MatrixTool::multiply()
     *
     * @param array $matrixArrA two dimension array within <b>float</b> type number
     * @param array $matrixArrB two dimension array within <b>float</b> type number
     * @param array $matrixArrC two dimension array within <b>float</b> type number
     * @param <b>float<b> $alpha
     * @param <b>float<b> $beta
     * @return array two dimension array within <b>float</b> type number
     * @throws \Exception
     */
    public function multiplyS( $matrixArrA, $matrixArrB, $matrixArrC = [][], $alpha = 1.0, $beta = 0.0 ){}
    
    /**
     * This function computes the dot product of vectors $oneDimensionArrA and $oneDimensionArrB
     * 
     * ∑ i = 1 n ( $oneDimensionArrA[ k ] * $oneDimensionArrB [ j ] ) 
     *      where 
     *          k = 1 + ( i - 1 ) * incx 
     *          and j = 1 + ( i - 1 ) * incy .
     * 
     * @param array $oneDimensionArrA one dimension array within double type number
     * @param array $oneDimensionArrB one dimension array within double type number
     * @param int $strideA
     * @param int $strideB
     * @return double 
     * @throws \Exception
     */
    public function dot( $oneDimensionArrA, $oneDimensionArrB, $strideA = 1, $strideB = 1 ){}
    
    /**
     * This function computes the dot product of vectors $oneDimensionArrA and $oneDimensionArrB
     * 
     * ∑ i = 1 n ( $oneDimensionArrA[ k ] * $oneDimensionArrB [ j ] ) 
     *      where 
     *          k = 1 + ( i - 1 ) * incx 
     *          and j = 1 + ( i - 1 ) * incy .
     * 
     * @see MatrixTool::dot()
     * 
     * @param array $oneDimensionArrA one dimension array within <b>float</b> type number
     * @param array $oneDimensionArrB one dimension array within <b>float</b> type number
     * @param int $strideA
     * @param int $strideB
     * @return float 
     * @throws \Exception
     */
    public function dotS( $oneDimensionArrA, $oneDimensionArrB, $strideA = 1, $strideB = 1 ){}
    
    /**
     * This function scales the vector $oneDimensionArrA multiply the scalar $alpha 
     * 
     *  $result[ j ] = $alpha * $oneDimensionArrA [ j ] 
     *      for i = 1 , … , n 
     *          and j = 1 + ( i - 1 ) *  $increase 
     *  
     * @param double $alpha
     * @param array $oneDimensionArrA one dimension array within double type number
     * @param int $increase
     * @return array one dimension array within double type number
     * @throws \Exception
     */
    public function scal( $alpha, $oneDimensionArrA, $increase = 1 ){}
    
    /**
     * This function scales the vector $oneDimensionArrA multiply the scalar $alpha 
     * 
     *  $result[ j ] = $alpha * $oneDimensionArrA [ j ] 
     *      for i = 1 , … , n 
     *          and j = 1 + ( i - 1 ) *  $increase 
     *  
     *  @see MatrixTool::scal()
     *  
     * @param float $alpha
     * @param array $oneDimensionArrA one dimension array within <b>float</b> type number
     * @param int $increase
     * @return array one dimension array within <b>float</b> type number
     * @throws \Exception
     */
    public function scalS( $alpha, $oneDimensionArrA, $increase = 1 ){}
    
    /**
     * This function finds the (smallest) index of the element of the maximum absolute value
     *      for i = 1 , … , n 
     *          and j = 1 + ( i - 1 ) *  $increase 
     * @param array $oneDimensionArrA one dimension array within double type number
     * @param int $increase only $increase ==1 is supported, something wrong with CUDA 
     * @return int return the index of the element of the maximum value. The first index of an array is 0.
     * @throws \Exception
     */
    public function amax( $oneDimensionArrA, $increase = 1 ){}
    
    /**
     * This function finds the (smallest) index of the element of the maximum value
     *      for i = 1 , … , n 
     *          and j = 1 + ( i - 1 ) *  $increase 
     * @param array $oneDimensionArrA one dimension array within <b>float</b> type number
     * @param int $increase only $increase ==1 is supported, something wrong with CUDA 
     * @return int return the index of the element of the maximum value. The first index of an array is 0.
     * @throws \Exception
     */
    public function amaxS( $oneDimensionArrA, $increase = 1 ){}
    
    /**
     * This function finds the (smallest) index of the element of the minimum absolute value
     *      for i = 1 , … , n 
     *          and j = 1 + ( i - 1 ) *  $increase 
     * @param array $oneDimensionArrA one dimension array within double type number
     * @param int $increase only $increase ==1 is supported, something wrong with CUDA 
     * @return int return the index of the element of the minimum value. The first index of an array is 0.
     * @throws \Exception
     */
    public function amin( $oneDimensionArrA, $increase = 1 ){}
    
    /**
     * This function finds the (smallest) index of the element of the minimum absolute value
     *      for i = 1 , … , n 
     *          and j = 1 + ( i - 1 ) *  $increase 
     * @param array $oneDimensionArrA one dimension array within <b>float</b> type number
     * @param int $increase only $increase ==1 is supported, something wrong with CUDA 
     * @return int return the index of the element of the minimum value. The first index of an array is 0.
     * @throws \Exception
     */
    public function aminS( $oneDimensionArrA, $increase = 1 ){}
    
    /**
     * This function multiplies the vector $oneDimensionArrA by the scalar $alpha and adds it to the vector $oneDimensionArrB
     *      $alpha * $oneDimensionArrA + $oneDimensionArrB
     * @param array $oneDimensionArrA one dimension array within double type number
     * @param array $oneDimensionArrB one dimension array within double type number
     * @param double $alpha
     * @param int $strideA
     * @param int $strideB
     * @return array one dimension array within double type number
     * @throws \Exception
     */
    public function axpy( $oneDimensionArrA, $oneDimensionArrB, $alpha = 1.0, $strideA = 1, $strideB = 1 ){}
    
    /**
     * This function multiplies the vector $oneDimensionArrA by the scalar $alpha and adds it to the vector $oneDimensionArrB
     *      $alpha * $oneDimensionArrA + $oneDimensionArrB
     * @param array $oneDimensionArrA one dimension array within <b>float</b>  type number
     * @param array $oneDimensionArrB one dimension array within <b>float</b>  type number
     * @param float $alpha
     * @param int $strideA
     * @param int $strideB
     * @return array one dimension array within <b>float</b>  type number
     * @throws \Exception
     */
    public function axpyS( $oneDimensionArrA, $oneDimensionArrB, $alpha = 1.0, $strideA = 1, $strideB = 1 ){}
    
    /**
     * This function performs the matrix-vector multiplication.The vector is column vector.
     *      $oneDimensionArrY = $alpha * $matrixArrA * $oneDimensionArrX + $beta * $oneDimensionArrY
     * 
     * @param array $matrixArrA two dimension array within double type number
     * @param array $oneDimensionArrX one dimension array within double type number
     * @param array $oneDimensionArrY one dimension array within double type number
     * @param double $alpha
     * @param double $beta
     * @param int $increaseX
     * @param int $increaseY
     */
    public function gemv( $matrixArrA, $oneDimensionArrX, $oneDimensionArrY = [], $alpha = 1.0, $beta = 1.0, $strideX = 1, $strideY = 1 ){}
    
    /**
     * This function performs the matrix-vector multiplication.The vector is column vector.
     *      $oneDimensionArrY = $alpha * $matrixArrA * $oneDimensionArrX + $beta * $oneDimensionArrY
     * 
     * @param array $matrixArrA two dimension array within double type number
     * @param array $oneDimensionArrX one dimension array within <b>float</b> type number
     * @param array $oneDimensionArrY one dimension array within <b>float</b> type number
     * @param float $alpha
     * @param float $beta
     * @param int $increaseX
     * @param int $increaseY
     */
    public function gemvS( $matrixArrA, $oneDimensionArrX, $oneDimensionArrY = [], $alpha = 1.0, $beta = 1.0, $strideX = 1, $strideY = 1 ){}
    
    
    
    
}


class Math{
    public static $DEVICE_ID;
    
    /**
     * getter
     * @return int
     */
    public static function getDeviceId(){}
    
    /**
     * setter
     * @param int $deviceId
     */
    public static function setDeviceId( $deviceId ){}
    
    /**
     * add $alpaha on each element of $arrA.
     * @param array $arr 1 or more dimension array within double type number
     * @param double $alpha
     */
    public static function arrayAdd( $arrA, $alpha){}
    
    //TODO basic operation
    public static function subtractArray( $numX, $arrA ){}
    
    //TODO basic operation
    public static function arrayMultiply( $arrA, $alpha){}
    
    //TODO basic operation
    public static function divideArray( $numX, $arrA ){}
    
    //TODO basic operation
    public static function arrayPower( $arrA, $alpha){}
    
    //TODO basic operation
    public static function arraySquareRoot( $arrA, $alpha){}
    
    //TODO basic operation
    public static function arrayCubeRoot( $arrA, $alpha){}
    
    //TODO basic operation
    public static function logEArray( $arrA ){}
    
    //TODO basic operation
    public static function log2Array( $arrA ){}
    
    //TODO basic operation
    public static function log10Array( $arrA ){}
    

    //-------------------
    
    /**
     * This functin caculate hadamard product.
     * result = $matrixA * $matrixB <br/>
     * [ [ 1, 2, 3], [ 4, 5, 6 ] ] * [ [ 1, 1, 1], [ 2, 2, 2 ] ]
     * = [ [ 1, 2, 3], [ 8, 10, 12 ] ]
     * 
     * @param array $matrixA two dimension array within double type number
     * @param array $matrixB two dimension array within double type number
     * @return array two dimension array within double type number
     */
    public static function hadamardProduct( $matrixA, $matrixB ){}
    
    //TODO advanced operation
    public static function transpose( $arrA ){}
    
    
    
    
    
    
    
}



?>
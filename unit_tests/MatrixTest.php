<?php
namespace BS\matrix;

include_once __DIR__ . '/../vendor/autoload.php';

use PHPUnit\Framework\TestCase;
class MatrixTest extends TestCase {
    
    private function randomArr( $height, $width, $divisor = 1000 ){
        $arr = Util::initArrayBySize($height);
        for($i = 0 ; $i < $height; $i++){
            $tempArr = Util::initArrayBySize($width);
            for ($j = 0 ; $j < $width; $j++){
                $tempArr[] = rand( 1, $divisor ) + rand( 1, $divisor ) / $divisor ;
            }
            $arr[] = $tempArr;
        }
        
        return $arr;
    }
    
    private function multiply( $a, $b ){
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
    
    public function testInitArrayBySize(){
        $arr = Util::initArrayBySize(9999999);
        $this->assertIsArray($arr);
    }
    
    public function testCudaGetDeviceCount(){
        $deviceCount = Util::cudaGetDeviceCount();
        $this->assertGreaterThanOrEqual( 0, $deviceCount );
    }
    
    public function testGetDeviceNameById(){
        
        $deviceId = 0;
        $name = Util::getDeviceNameById($deviceId);
        $this->assertIsString($name);
        
        $deviceId = -1;
        $this->expectException(\Exception::class);
        $name = Util::getDeviceNameById($deviceId);
        
    }
    
    //----------------------------
    
    
    public function testBLASConstructor(){
        $BLAS = new BLAS();
        $this->assertInstanceOf( BLAS::class, $BLAS );
        
        return $BLAS;
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testGetAndSetHandle( BLAS $BLAS ){
        
        $handle = $BLAS->getHandle();
        $this->assertIsResource($handle);
        
        $BLAS->setHandle($handle);
        
        $handle = $BLAS->getHandle();
        $this->assertIsResource($handle);
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testMultiply( BLAS $BLAS ){ 
        $matrixArrA = $this->randomArr( 100, 200 );
        $matrixArrB = $this->randomArr( 200, 150 );
        $matrixArrC = $this->randomArr( 100, 150 );
                                        
        $cpuResultArr = $this->multiply( $matrixArrA, $matrixArrB );
        
        $gpuResultArr = $BLAS->multiply( $matrixArrA, $matrixArrB, $matrixArrC, 1.0, 0.0 );
        
        $this->assertEquals( round( $cpuResultArr[0][0], 5 ), round( $gpuResultArr[0][0], 5 ) ) ;
        $this->assertEquals( round( $cpuResultArr[11][11], 5 ), round( $gpuResultArr[11][11], 5 ) );
        $this->assertEquals( round( $cpuResultArr[22][22], 5 ), round( $gpuResultArr[22][22], 5 ) );
        $this->assertEquals( round( $cpuResultArr[33][33], 5 ), round( $gpuResultArr[33][33], 5 ) );
        $this->assertEquals( round( $cpuResultArr[44][44], 5 ), round( $gpuResultArr[44][44], 5 ) );
        $this->assertEquals( round( $cpuResultArr[99][149], 5 ), round( $gpuResultArr[99][149], 5 ) );
        
        $tempGpuResultArr = $BLAS->multiply( $matrixArrA, $matrixArrB, $matrixArrC, 2.6, 4.1 );
        $this->assertEquals( 
                round( $gpuResultArr[66][66] * 2.6 + 4.1 * $matrixArrC[66][66], 6 ), 
                round( $tempGpuResultArr[66][66], 6 ) 
        );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testMultiplyS( BLAS $BLAS ){ 
        $matrixArrA = $this->randomArr( 100, 200 , 10 );
        $matrixArrB = $this->randomArr( 200, 150 , 10 );
        $matrixArrC = $this->randomArr( 100, 150 , 10 );
                                        
        $cpuResultArr = $this->multiply( $matrixArrA, $matrixArrB );
        
        $gpuResultArr = $BLAS->MultiplyS( $matrixArrA, $matrixArrB, $matrixArrC, 1.0, 0.0 );
        
        $this->assertEquals( round( $cpuResultArr[0][0], 2 ), round( $gpuResultArr[0][0], 2 ) ) ;
        $this->assertEquals( round( $cpuResultArr[11][11], 2 ), round( $gpuResultArr[11][11], 2 ) );
        $this->assertEquals( round( $cpuResultArr[22][22], 2 ), round( $gpuResultArr[22][22], 2 ) );
        $this->assertEquals( round( $cpuResultArr[33][33], 2 ), round( $gpuResultArr[33][33], 2 ) );
        $this->assertEquals( round( $cpuResultArr[44][44], 2 ), round( $gpuResultArr[44][44], 2 ) );
        $this->assertEquals( round( $cpuResultArr[99][149], 2 ), round( $gpuResultArr[99][149], 2 ) );
        
        $tempGpuResultArr = $BLAS->multiply( $matrixArrA, $matrixArrB, $matrixArrC, 2.6, 4.1 );
        $this->assertEquals( 
                round( $gpuResultArr[66][66] * 2.6 + 4.1 * $matrixArrC[66][66], 2 ), 
                round( $tempGpuResultArr[66][66], 2 ) 
        );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testDot( BLAS $BLAS ){
        $matrixArrA = [ 1.1, 2, 3, 4 ];
        $matrixArrB = [ 1,   2, 3, 4, 5, 6, 7.7, 8 ];
        
        $res = $BLAS->dot( $matrixArrA, $matrixArrB, 1, 2 );
        
        $this->assertEquals( 1.1 * 1 + 2 * 3 + 3 * 5 + 4 * 7.7, $res );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testDotS( BLAS $BLAS ){
        $matrixArrA = [ 1.1, 2, 3, 4 ];
        $matrixArrB = [ 1,   2, 3, 4, 5, 6, 7.7, 8 ];
        
        $res = $BLAS->dotS( $matrixArrA, $matrixArrB, 1, 2 );
        
        $this->assertEquals( 
                round( 1.1 * 1 + 2 * 3 + 3 * 5 + 4 * 7.7, 5 ), 
                round( $res, 5 )
        );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testScal( BLAS $BLAS ){
        $matrixArrA = [ 1.1, 2, 3, 4 ];
        $resArr = $BLAS->scal( 2.1, $matrixArrA );
        
        $this->assertEquals( 
                array_map( function($value){ return 2.1 * $value;}, $matrixArrA ), 
                $resArr 
        );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testScalS( BLAS $BLAS ){
        $matrixArrA = [ 1.1, 2, 3, 4 ];
        $resArr = $BLAS->scalS( 2.1, $matrixArrA );
        
        $this->assertEquals( round( 1.1 * 2.1, 2 ), round( $resArr[0], 2 ) );
        $this->assertEquals( round( 2 * 2.1, 2 ), round( $resArr[1], 2 ) );
        $this->assertEquals( round( 3 * 2.1, 2 ), round( $resArr[2], 2 ) );
        $this->assertEquals( round( 4 * 2.1, 2 ), round( $resArr[3], 2 ) );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testAmax( BLAS $BLAS ){
        
        $oneDimensionArrA = [ 1.0, 2.0, 3.0, 44.0, 5.0 ];
        $res = $BLAS->amax($oneDimensionArrA);
        $this->assertEquals(3, $res);
        
        $oneDimensionArrA = [ 1.0, -2.0, 3.0, -44.0, 5.0 ];
        $res = $BLAS->amax($oneDimensionArrA);
        $this->assertEquals(3, $res);
        
//         $oneDimensionArrA = [ 1.0, 2.0, 3.0, 4.0, 5.0 ];
//         $res = $BLAS->amax( $oneDimensionArrA, 2 );var_dump($res);//something wrong with CUDA
//         $this->assertEquals(4, $res);var_dump($res);
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testAmaxS( BLAS $BLAS ){
        
        $oneDimensionArrA = [ 1.0, 2.0, 3.0, 44.0, 5.0 ];
        $res = $BLAS->amaxS($oneDimensionArrA);
        $this->assertEquals( 3, $res );
        
        $oneDimensionArrA = [ 1.0, -2.0, 3.0, -44.0, 5.0 ];
        $res = $BLAS->amaxS($oneDimensionArrA);
        $this->assertEquals( 3, $res );
        
//         $oneDimensionArrA = [ 1.0, 2.0, 3.0, 4.0, 5.0 ];
//         $res = $BLAS->amaxS( $oneDimensionArrA, 2 );//something wrong with CUDA
//         $this->assertEquals( 4, $res );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testAmin( BLAS $BLAS ){
        
        $oneDimensionArrA = [ 1.0, 2.0, 3.0, 0.04, 5.0 ];
        $res = $BLAS->amin($oneDimensionArrA);
        $this->assertEquals( 3, $res );
        
        $oneDimensionArrA = [ 1.0, -2.0, 3.0, -0.04, 5.0 ];
        $res = $BLAS->amin($oneDimensionArrA);
        $this->assertEquals( 3, $res );
        
//         $oneDimensionArrA = [ 1.0, 2.0, 3.0, 0.04, 0.04 ];
//         $res = $BLAS->amin( $oneDimensionArrA, 2 );//something wrong with CUDA
//         $this->assertEquals( 4, $res );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testAminS( BLAS $BLAS ){
        
        $oneDimensionArrA = [ 1.0, 2.0, 3.0, 0.04, 5.0 ];
        $res = $BLAS->aminS($oneDimensionArrA);
        $this->assertEquals( 3, $res );
        
        $oneDimensionArrA = [ 1.0, -2.0, 3.0, -0.04, 5.0 ];
        $res = $BLAS->aminS($oneDimensionArrA);
        $this->assertEquals( 3, $res );
        
//         $oneDimensionArrA = [ 1.0, 2.0, 0.03];
//         $res = $BLAS->aminS( $oneDimensionArrA, 2);//something wrong with CUDA
//         $this->assertEquals( 2, $res );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testAxpy( BLAS $BLAS ){
        
        $oneDimensionArrA = [ 1.0, 2.0, 3.0, 4.0, 5.0 ];
        $oneDimensionArrB = [ 6.0, 7.0, 8.0, 9.0, 10.0 ];
        $resArr = $BLAS->axpy( $oneDimensionArrA, $oneDimensionArrB );
        $this->assertEquals( [ 7.0, 9.0, 11.0, 13.0, 15.0 ], $resArr );
        
        $resArr = $BLAS->axpy( $oneDimensionArrA, $oneDimensionArrB, 2.1, 1, 1 );
        $this->assertEquals( [ 8.1, 11.2, 14.3, 17.4, 20.5  ], $resArr );
        
        $resArr = $BLAS->axpy( $oneDimensionArrA, $oneDimensionArrB, 2.1, 2, 2 );
        $this->assertEquals( [ 8.1, 7.0, 14.3, 9.0, 20.5  ], $resArr );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testAxpyS( BLAS $BLAS ){
        
        $oneDimensionArrA = [ 1.0, 2.0, 3.0, 4.0, 5.0 ];
        $oneDimensionArrB = [ 6.0, 7.0, 8.0, 9.0, 10.0 ];
        $resArr = $BLAS->axpyS( $oneDimensionArrA, $oneDimensionArrB );
        $this->assertEquals( [ 7.0, 9.0, 11.0, 13.0, 15.0 ], $resArr );
        
        $resArr = $BLAS->axpyS( $oneDimensionArrA, $oneDimensionArrB, 2.1, 1, 1 );
        $resArr = array_map(function($value){return round( $value, 1 );}, $resArr);
        $this->assertEquals( [ 8.1, 11.2, 14.3, 17.4, 20.5  ], $resArr );
        
        $resArr = $BLAS->axpyS( $oneDimensionArrA, $oneDimensionArrB, 2.1, 2, 2 );
        $resArr = array_map(function($value){return round( $value, 1 );}, $resArr);
        $this->assertEquals( [ 8.1, 7.0, 14.3, 9.0, 20.5  ], $resArr );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testGemv( BLAS $BLAS ){
        
        $matrixArrA = [
                [ 1.0, 2.0, 3.0, ],
                [ 4.0, 5.0, 6.0, ],
        ];
 
        $oneDimensionArrX = [ 1.0, 2.0, 3.0, 1, 1, 1 ];
        $resArr = $BLAS->gemv( $matrixArrA, $oneDimensionArrX );
        $this->assertEquals( [ 9, 12, 15 ], $resArr );
        
        $oneDimensionArrY = [ 1.0, 1.0, 1.0, 2.0, 2.0, 2.0 ];
        $resArr = $BLAS->gemv( $matrixArrA, $oneDimensionArrX, $oneDimensionArrY, 2.0, 2.0, 2.0, 2.0 );
        $this->assertEquals( [ 28, 1, 36, 2, 46, 2 ], $resArr );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testGemvS( BLAS $BLAS ){
        
        $matrixArrA = [
                [ 1.0, 2.0, 3.0, ],
                [ 4.0, 5.0, 6.0, ],
        ];
 
        $oneDimensionArrX = [ 1.0, 2.0, 3.0, 1, 1, 1 ];
        $resArr = $BLAS->gemvS( $matrixArrA, $oneDimensionArrX );
        $this->assertEquals( [ 9, 12, 15 ], $resArr );
        
        $oneDimensionArrY = [ 1.0, 1.0, 1.0, 2.0, 2.0, 2.0 ];
        $resArr = $BLAS->gemvS( $matrixArrA, $oneDimensionArrX, $oneDimensionArrY, 2.0, 2.0, 2.0, 2.0 );
        $this->assertEquals( [ 28, 1, 36, 2, 46, 2 ], $resArr );
        
    }
    
    
    
}


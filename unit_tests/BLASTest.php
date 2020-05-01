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
        $blas = new BLAS();
        $this->assertInstanceOf( BLAS::class, $blas );
        
        return $blas;
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testGetAndSetHandle( BLAS $blas ){
        
        $handle = $blas->getHandle();
        $this->assertIsResource($handle);
        
        $blas->setHandle($handle);
        
        $handle = $blas->getHandle();
        $this->assertIsResource($handle);
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testMultiply( BLAS $blas ){ 
        $matrixArrA = $this->randomArr( 100, 200 );
        $matrixArrB = $this->randomArr( 200, 150 );
        $matrixArrC = $this->randomArr( 100, 150 );
                                        
        $cpuResultArr = $this->multiply( $matrixArrA, $matrixArrB );
        
        $gpuResultArr = $blas->multiply( $matrixArrA, $matrixArrB, $matrixArrC, 1.0, 0.0 );
        
        $this->assertEquals( round( $cpuResultArr[0][0], 5 ), round( $gpuResultArr[0][0], 5 ) ) ;
        $this->assertEquals( round( $cpuResultArr[11][11], 5 ), round( $gpuResultArr[11][11], 5 ) );
        $this->assertEquals( round( $cpuResultArr[22][22], 5 ), round( $gpuResultArr[22][22], 5 ) );
        $this->assertEquals( round( $cpuResultArr[33][33], 5 ), round( $gpuResultArr[33][33], 5 ) );
        $this->assertEquals( round( $cpuResultArr[44][44], 5 ), round( $gpuResultArr[44][44], 5 ) );
        $this->assertEquals( round( $cpuResultArr[99][149], 5 ), round( $gpuResultArr[99][149], 5 ) );
        
        $tempGpuResultArr = $blas->multiply( $matrixArrA, $matrixArrB, $matrixArrC, 2.6, 4.1 );
        $this->assertEquals( 
                round( $gpuResultArr[66][66] * 2.6 + 4.1 * $matrixArrC[66][66], 6 ), 
                round( $tempGpuResultArr[66][66], 6 ) 
        );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testMultiplyS( BLAS $blas ){ 
        $matrixArrA = $this->randomArr( 100, 200 , 10 );
        $matrixArrB = $this->randomArr( 200, 150 , 10 );
        $matrixArrC = $this->randomArr( 100, 150 , 10 );
                                        
        $cpuResultArr = $this->multiply( $matrixArrA, $matrixArrB );
        
        $gpuResultArr = $blas->MultiplyS( $matrixArrA, $matrixArrB, $matrixArrC, 1.0, 0.0 );
        
        $this->assertEquals( round( $cpuResultArr[0][0], 2 ), round( $gpuResultArr[0][0], 2 ) ) ;
        $this->assertEquals( round( $cpuResultArr[11][11], 2 ), round( $gpuResultArr[11][11], 2 ) );
        $this->assertEquals( round( $cpuResultArr[22][22], 2 ), round( $gpuResultArr[22][22], 2 ) );
        $this->assertEquals( round( $cpuResultArr[33][33], 2 ), round( $gpuResultArr[33][33], 2 ) );
        $this->assertEquals( round( $cpuResultArr[44][44], 2 ), round( $gpuResultArr[44][44], 2 ) );
        $this->assertEquals( round( $cpuResultArr[99][149], 2 ), round( $gpuResultArr[99][149], 2 ) );
        
        $tempGpuResultArr = $blas->multiply( $matrixArrA, $matrixArrB, $matrixArrC, 2.6, 4.1 );
        $this->assertEquals( 
                round( $gpuResultArr[66][66] * 2.6 + 4.1 * $matrixArrC[66][66], 2 ), 
                round( $tempGpuResultArr[66][66], 2 ) 
        );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testDot( BLAS $blas ){
        $matrixArrA = [ 1.1, 2, 3, 4 ];
        $matrixArrB = [ 1,   2, 3, 4, 5, 6, 7.7, 8 ];
        
        $res = $blas->dot( $matrixArrA, $matrixArrB, 1, 2 );
        
        $this->assertEquals( 1.1 * 1 + 2 * 3 + 3 * 5 + 4 * 7.7, $res );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testDotS( BLAS $blas ){
        $matrixArrA = [ 1.1, 2, 3, 4 ];
        $matrixArrB = [ 1,   2, 3, 4, 5, 6, 7.7, 8 ];
        
        $res = $blas->dotS( $matrixArrA, $matrixArrB, 1, 2 );
        
        $this->assertEquals( 
                round( 1.1 * 1 + 2 * 3 + 3 * 5 + 4 * 7.7, 5 ), 
                round( $res, 5 )
        );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testScal( BLAS $blas ){
        $matrixArrA = [ 1.1, 2, 3, 4 ];
        $resArr = $blas->scal( 2.1, $matrixArrA );
        
        $this->assertEquals( 
                array_map( function($value){ return 2.1 * $value;}, $matrixArrA ), 
                $resArr 
        );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testScalS( BLAS $blas ){
        $matrixArrA = [ 1.1, 2, 3, 4 ];
        $resArr = $blas->scalS( 2.1, $matrixArrA );
        
        $this->assertEquals( round( 1.1 * 2.1, 2 ), round( $resArr[0], 2 ) );
        $this->assertEquals( round( 2 * 2.1, 2 ), round( $resArr[1], 2 ) );
        $this->assertEquals( round( 3 * 2.1, 2 ), round( $resArr[2], 2 ) );
        $this->assertEquals( round( 4 * 2.1, 2 ), round( $resArr[3], 2 ) );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testAmax( BLAS $blas ){
        
        $oneDimensionArrA = [ 1.0, 2.0, 3.0, 44.0, 5.0 ];
        $res = $blas->amax($oneDimensionArrA);
        $this->assertEquals(3, $res);
        
        $oneDimensionArrA = [ 1.0, -2.0, 3.0, -44.0, 5.0 ];
        $res = $blas->amax($oneDimensionArrA);
        $this->assertEquals(3, $res);
        
//         $oneDimensionArrA = [ 1.0, 2.0, 3.0, 4.0, 5.0 ];
//         $res = $blas->amax( $oneDimensionArrA, 2 );var_dump($res);//something wrong with CUDA
//         $this->assertEquals(4, $res);var_dump($res);
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testAmaxS( BLAS $blas ){
        
        $oneDimensionArrA = [ 1.0, 2.0, 3.0, 44.0, 5.0 ];
        $res = $blas->amaxS($oneDimensionArrA);
        $this->assertEquals( 3, $res );
        
        $oneDimensionArrA = [ 1.0, -2.0, 3.0, -44.0, 5.0 ];
        $res = $blas->amaxS($oneDimensionArrA);
        $this->assertEquals( 3, $res );
        
//         $oneDimensionArrA = [ 1.0, 2.0, 3.0, 4.0, 5.0 ];
//         $res = $blas->amaxS( $oneDimensionArrA, 2 );//something wrong with CUDA
//         $this->assertEquals( 4, $res );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testAmin( BLAS $blas ){
        
        $oneDimensionArrA = [ 1.0, 2.0, 3.0, 0.04, 5.0 ];
        $res = $blas->amin($oneDimensionArrA);
        $this->assertEquals( 3, $res );
        
        $oneDimensionArrA = [ 1.0, -2.0, 3.0, -0.04, 5.0 ];
        $res = $blas->amin($oneDimensionArrA);
        $this->assertEquals( 3, $res );
        
//         $oneDimensionArrA = [ 1.0, 2.0, 3.0, 0.04, 0.04 ];
//         $res = $blas->amin( $oneDimensionArrA, 2 );//something wrong with CUDA
//         $this->assertEquals( 4, $res );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testAminS( BLAS $blas ){
        
        $oneDimensionArrA = [ 1.0, 2.0, 3.0, 0.04, 5.0 ];
        $res = $blas->aminS($oneDimensionArrA);
        $this->assertEquals( 3, $res );
        
        $oneDimensionArrA = [ 1.0, -2.0, 3.0, -0.04, 5.0 ];
        $res = $blas->aminS($oneDimensionArrA);
        $this->assertEquals( 3, $res );
        
//         $oneDimensionArrA = [ 1.0, 2.0, 0.03];
//         $res = $blas->aminS( $oneDimensionArrA, 2);//something wrong with CUDA
//         $this->assertEquals( 2, $res );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testAxpy( BLAS $blas ){
        
        $oneDimensionArrA = [ 1.0, 2.0, 3.0, 4.0, 5.0 ];
        $oneDimensionArrB = [ 6.0, 7.0, 8.0, 9.0, 10.0 ];
        $resArr = $blas->axpy( $oneDimensionArrA, $oneDimensionArrB );
        $this->assertEquals( [ 7.0, 9.0, 11.0, 13.0, 15.0 ], $resArr );
        
        $resArr = $blas->axpy( $oneDimensionArrA, $oneDimensionArrB, 2.1, 1, 1 );
        $this->assertEquals( [ 8.1, 11.2, 14.3, 17.4, 20.5  ], $resArr );
        
        $resArr = $blas->axpy( $oneDimensionArrA, $oneDimensionArrB, 2.1, 2, 2 );
        $this->assertEquals( [ 8.1, 7.0, 14.3, 9.0, 20.5  ], $resArr );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testAxpyS( BLAS $blas ){
        
        $oneDimensionArrA = [ 1.0, 2.0, 3.0, 4.0, 5.0 ];
        $oneDimensionArrB = [ 6.0, 7.0, 8.0, 9.0, 10.0 ];
        $resArr = $blas->axpyS( $oneDimensionArrA, $oneDimensionArrB );
        $this->assertEquals( [ 7.0, 9.0, 11.0, 13.0, 15.0 ], $resArr );
        
        $resArr = $blas->axpyS( $oneDimensionArrA, $oneDimensionArrB, 2.1, 1, 1 );
        $resArr = array_map(function($value){return round( $value, 1 );}, $resArr);
        $this->assertEquals( [ 8.1, 11.2, 14.3, 17.4, 20.5  ], $resArr );
        
        $resArr = $blas->axpyS( $oneDimensionArrA, $oneDimensionArrB, 2.1, 2, 2 );
        $resArr = array_map(function($value){return round( $value, 1 );}, $resArr);
        $this->assertEquals( [ 8.1, 7.0, 14.3, 9.0, 20.5  ], $resArr );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testGemv( BLAS $blas ){
        
        $matrixArrA = [
                [ 1.0, 2.0, 3.0, ],
                [ 4.0, 5.0, 6.0, ],
        ];
 
        $oneDimensionArrX = [ 1.0, 2.0, 3.0, 1, 1, 1 ];
        $resArr = $blas->gemv( $matrixArrA, $oneDimensionArrX );
        $this->assertEquals( [ 9, 12, 15 ], $resArr );
        
        $oneDimensionArrY = [ 1.0, 1.0, 1.0, 2.0, 2.0, 2.0 ];
        $resArr = $blas->gemv( $matrixArrA, $oneDimensionArrX, $oneDimensionArrY, 2.0, 2.0, 2.0, 2.0 );
        $this->assertEquals( [ 28, 1, 36, 2, 46, 2 ], $resArr );
        
    }
    
    /**
     * @depends testBLASConstructor
     */
    public function testGemvS( BLAS $blas ){
        
        $matrixArrA = [
                [ 1.0, 2.0, 3.0, ],
                [ 4.0, 5.0, 6.0, ],
        ];
 
        $oneDimensionArrX = [ 1.0, 2.0, 3.0, 1, 1, 1 ];
        $resArr = $blas->gemvS( $matrixArrA, $oneDimensionArrX );
        $this->assertEquals( [ 9, 12, 15 ], $resArr );
        
        $oneDimensionArrY = [ 1.0, 1.0, 1.0, 2.0, 2.0, 2.0 ];
        $resArr = $blas->gemvS( $matrixArrA, $oneDimensionArrX, $oneDimensionArrY, 2.0, 2.0, 2.0, 2.0 );
        $this->assertEquals( [ 28, 1, 36, 2, 46, 2 ], $resArr );
        
    }
    
    
    
    
    
}


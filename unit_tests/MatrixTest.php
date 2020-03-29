<?php
namespace BS;

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
    
    
    public function testMatrixToolConstructor(){
        $matrixTool = new MatrixTool();
        $this->assertInstanceOf( MatrixTool::class, $matrixTool );
        
        return $matrixTool;
    }
    
    /**
     * @depends testMatrixToolConstructor
     */
    public function testGetAndSetHandle( MatrixTool $matrixTool ){
        
        $handle = $matrixTool->getHandle();
        $this->assertIsResource($handle);
        
        $matrixTool->setHandle($handle);
        
        $handle = $matrixTool->getHandle();
        $this->assertIsResource($handle);
        
    }
    
    /**
     * @depends testMatrixToolConstructor
     */
    public function testMultiply( MatrixTool $matrixTool ){ 
        $matrixArrA = $this->randomArr( 100, 200 );
        $matrixArrB = $this->randomArr( 200, 150 );
        $matrixArrC = $this->randomArr( 100, 150 );
                                        
        $cpuResultArr = $this->multiply( $matrixArrA, $matrixArrB );
        
        $gpuResultArr = $matrixTool->multiply( $matrixArrA, $matrixArrB, $matrixArrC, 1.0, 0.0 );
        
        $this->assertEquals( round( $cpuResultArr[0][0], 5 ), round( $gpuResultArr[0][0], 5 ) ) ;
        $this->assertEquals( round( $cpuResultArr[11][11], 5 ), round( $gpuResultArr[11][11], 5 ) );
        $this->assertEquals( round( $cpuResultArr[22][22], 5 ), round( $gpuResultArr[22][22], 5 ) );
        $this->assertEquals( round( $cpuResultArr[33][33], 5 ), round( $gpuResultArr[33][33], 5 ) );
        $this->assertEquals( round( $cpuResultArr[44][44], 5 ), round( $gpuResultArr[44][44], 5 ) );
        $this->assertEquals( round( $cpuResultArr[99][149], 5 ), round( $gpuResultArr[99][149], 5 ) );
        
        $tempGpuResultArr = $matrixTool->multiply( $matrixArrA, $matrixArrB, $matrixArrC, 2.6, 4.1 );
        $this->assertEquals( 
                round( $gpuResultArr[66][66] * 2.6 + 4.1 * $matrixArrC[66][66], 6 ), 
                round( $tempGpuResultArr[66][66], 6 ) 
        );
        
    }
    
    /**
     * @depends testMatrixToolConstructor
     */
    public function testMultiplyS( MatrixTool $matrixTool ){ 
        $matrixArrA = $this->randomArr( 100, 200 , 10 );
        $matrixArrB = $this->randomArr( 200, 150 , 10 );
        $matrixArrC = $this->randomArr( 100, 150 , 10 );
                                        
        $cpuResultArr = $this->multiply( $matrixArrA, $matrixArrB );
        
        $gpuResultArr = $matrixTool->MultiplyS( $matrixArrA, $matrixArrB, $matrixArrC, 1.0, 0.0 );
        
        $this->assertEquals( round( $cpuResultArr[0][0], 2 ), round( $gpuResultArr[0][0], 2 ) ) ;
        $this->assertEquals( round( $cpuResultArr[11][11], 2 ), round( $gpuResultArr[11][11], 2 ) );
        $this->assertEquals( round( $cpuResultArr[22][22], 2 ), round( $gpuResultArr[22][22], 2 ) );
        $this->assertEquals( round( $cpuResultArr[33][33], 2 ), round( $gpuResultArr[33][33], 2 ) );
        $this->assertEquals( round( $cpuResultArr[44][44], 2 ), round( $gpuResultArr[44][44], 2 ) );
        $this->assertEquals( round( $cpuResultArr[99][149], 2 ), round( $gpuResultArr[99][149], 2 ) );
        
        $tempGpuResultArr = $matrixTool->multiply( $matrixArrA, $matrixArrB, $matrixArrC, 2.6, 4.1 );
        $this->assertEquals( 
                round( $gpuResultArr[66][66] * 2.6 + 4.1 * $matrixArrC[66][66], 2 ), 
                round( $tempGpuResultArr[66][66], 2 ) 
        );
        
    }
    
    /**
     * @depends testMatrixToolConstructor
     */
    public function testDot( MatrixTool $matrixTool ){
        $matrixArrA = [ 1.1, 2, 3, 4 ];
        $matrixArrB = [ 1,   2, 3, 4, 5, 6, 7.7, 8 ];
        
        $res = $matrixTool->dot( $matrixArrA, $matrixArrB, 1, 2 );
        
        $this->assertEquals( 1.1 * 1 + 2 * 3 + 3 * 5 + 4 * 7.7, $res );
        
    }
    
    /**
     * @depends testMatrixToolConstructor
     */
    public function testDotS( MatrixTool $matrixTool ){
        $matrixArrA = [ 1.1, 2, 3, 4 ];
        $matrixArrB = [ 1,   2, 3, 4, 5, 6, 7.7, 8 ];
        
        $res = $matrixTool->dotS( $matrixArrA, $matrixArrB, 1, 2 );
        
        $this->assertEquals( 
                round( 1.1 * 1 + 2 * 3 + 3 * 5 + 4 * 7.7, 5 ), 
                round( $res, 5 )
        );
        
    }
    
    /**
     * @depends testMatrixToolConstructor
     */
    public function testScal( MatrixTool $matrixTool ){
        $matrixArrA = [ 1.1, 2, 3, 4 ];
        $resArr = $matrixTool->scal( 2.1, $matrixArrA );
        
        $this->assertEquals( 
                array_map( function($value){ return 2.1 * $value;}, $matrixArrA ), 
                $resArr 
        );
        
    }
    
    /**
     * @depends testMatrixToolConstructor
     */
    public function testScalS( MatrixTool $matrixTool ){
        $matrixArrA = [ 1.1, 2, 3, 4 ];
        $resArr = $matrixTool->scalS( 2.1, $matrixArrA );
        
        $this->assertEquals( round( 1.1 * 2.1, 2 ), round( $resArr[0], 2 ) );
        $this->assertEquals( round( 2 * 2.1, 2 ), round( $resArr[1], 2 ) );
        $this->assertEquals( round( 3 * 2.1, 2 ), round( $resArr[2], 2 ) );
        $this->assertEquals( round( 4 * 2.1, 2 ), round( $resArr[3], 2 ) );
        
    }
    
    
    
}


<?php
namespace BS\matrix;

include_once __DIR__ . '/../vendor/autoload.php';

use PHPUnit\Framework\TestCase;
class MathTest extends TestCase {
    
    public function testGetDeviceId(){
        $res = Math::getDeviceId();
        
        $this->assertEquals( 0, $res );
        $this->assertEquals( Math::$DEVICE_ID, $res );
        
    }
    
    public function testSetDeviceId(){
        Math::setDeviceId(1);
        
        $res = Math::getDeviceId();
        $this->assertEquals( 1, $res );
        
        Math::setDeviceId(0);
    }
    
    public function testArrayAdd(){
        $arr = [ 1, 2, 3 ];
        $resArr = Math::arrayAdd( $arr, 1 );
        $this->assertEquals( [ 2, 3, 4 ], $resArr );
        
        //
        $arr = [
            [ 1, 2, 3 ],
            [ 4, 5, 6 ],
        ];
        $resArr = Math::arrayAdd( $arr, 10 );
        $this->assertEquals( 
                [
                        [ 11, 12, 13 ],
                        [ 14, 15, 16 ],
                ], $resArr 
        );
        
        //
        $arr = [
            [ 
                [ 1, 2, 3 ], 
                [ 4, 5, 6 ] 
            ],
            [ 
                [ 7, 8, 9 ], 
                [ 10, 11, 12 ] 
                
            ],
            
        ];
        $resArr = Math::arrayAdd( $arr, 1 );
        $this->assertEquals( 
            [
                [
                    [ 2, 3, 4 ], 
                    [ 5, 6, 7 ]
                ],
                [
                    [ 8, 9, 10 ], 
                    [ 11, 12, 13 ]
                ],
                
            ], $resArr 
        );
        
        
    }
    
    public function testSubtractArray(){
        $arr = [ 1, 2, 3 ];
        $resArr = Math::subtractArray( 1, $arr );
        $this->assertEquals( [ 0, -1, -2 ], $resArr );
        
        //
        $arr = [
            [ 1, 2, 3 ],
            [ 4, 5, 6 ],
        ];
        $resArr = Math::subtractArray( 10, $arr );
        $this->assertEquals( 
                [
                        [ 9, 8, 7 ],
                        [ 6, 5, 4 ],
                ], $resArr 
        );
        
        //
        $arr = [
            [ 
                [ 1, 2, 3 ], 
                [ 4, 5, 6 ] 
            ],
            [ 
                [ 7, 8, 9 ], 
                [ 10, 11, 12 ] 
                
            ],
            
        ];
        $resArr = Math::subtractArray( 10, $arr );
        $this->assertEquals( 
            [
                [
                    [ 9, 8, 7 ], 
                    [ 6, 5, 4 ]
                ],
                [
                    [ 3, 2, 1 ], 
                    [ 0, -1, -2 ]
                ],
                
            ], $resArr 
        );
        
        
    }
    
    public function testArrayMultiply(){
        $arr = [ 1, 2, 3 ];
        $resArr = Math::arrayMultiply( $arr, 2 );
        $this->assertEquals( [ 2, 4, 6 ], $resArr );
        
        //
        $arr = [
            [ 1, 2, 3 ],
            [ 4, 5, 6 ],
        ];
        $resArr = Math::arrayMultiply( $arr, 10 );
        $this->assertEquals( 
                [
                        [ 10, 20, 30 ],
                        [ 40, 50, 60 ],
                ], $resArr 
        );
        
        //
        $arr = [
            [ 
                [ 1, 2, 3 ], 
                [ 4, 5, 6 ] 
            ],
            [ 
                [ 7, 8, 9 ], 
                [ 10, 11, 12 ] 
                
            ],
            
        ];
        $resArr = Math::arrayMultiply( $arr, -1 );
        $this->assertEquals( 
            [
                [
                    [ -1, -2, -3 ],
                    [ -4, -5, -6 ]
                ],
                [
                    [ -7, -8, -9 ],
                    [ -10, -11, -12 ]
                    
                ],
                
            ], $resArr 
        );
        
        
    }
    
    public function testDivideArray(){
        $arr = [ 1, 2, 3 ];
        $resArr = Math::divideArray( 1, $arr );
        $resArr = array_map( function( $value ){ return round( $value, 2 ); }, $resArr );
        $this->assertEquals( [ 1, 0.5, 0.33 ], $resArr );
        
        //
        $arr = [
            [ 1, 2, 5 ],
            [ 5, 2, 1 ],
        ];
        $resArr = Math::divideArray( 10, $arr );
        
        $this->assertEquals( 
                [
                        [ 10, 5, 2 ],
                        [ 2, 5, 10 ],
                ], $resArr 
        );
        
        //
        $arr = [
            [ 
                [ 1, 2, 5 ], 
                [ 10, 20, 25 ] 
            ],
            [ 
                [ -1, -2, -5 ],
                [ -10, -20, -25 ] 
                
            ],
            
        ];
        $resArr = Math::divideArray( 100, $arr );
        $this->assertEquals( 
            [
                [
                    [ 100, 50, 20 ], 
                    [ 10, 5, 4 ]
                ],
                [
                    [ -100, -50, -20 ],
                    [ -10, -5, -4 ]
                ],
                
            ], $resArr 
        );
        
        
    }
    
    public function testArrayPower(){
        $arr = [ 1, 2, 3 ];
        $resArr = Math::arrayPower( $arr, 2 );
        $this->assertEquals( [ 1, 4, 9 ], $resArr );
    }
    
    public function testArraySquareRoot(){
        $arr = [ 1, 4, 9 ];
        $resArr = Math::arraySquareRoot( $arr );
        $this->assertEquals( [ 1, 2, 3 ], $resArr );
    }
    
    public function testArrayCubeRoot(){
        $arr = [ 1, 8, 27 ];
        $resArr = Math::arrayCubeRoot( $arr );
        $this->assertEquals( [ 1, 2, 3 ], $resArr );
    }
    
    public function testLogEArray(){
        $arr = [ 1, 8, 27 ];
        $resArr = Math::logEArray( $arr );
        $resArr = array_map( function( $value ){ return round( $value, 4 ); }, $resArr );
        $this->assertEquals( [ 0, 2.0794, 3.2958 ], $resArr );
    }
    
    public function testLog2Array(){
        $arr = [ 1, 4, 8 ];
        $resArr = Math::log2Array( $arr );
        $this->assertEquals( [ 0, 2, 3 ], $resArr );
    }
    
    public function testLog10Array(){
        $arr = [ 1, 100, 1000 ];
        $resArr = Math::log10Array( $arr );
        $this->assertEquals( [ 0, 2, 3 ], $resArr );
    }
    
    public function testHadamardProduct(){
        
        //
        $arrA = [
            [ 1, 2, 3 ],
            [ 4, 5, 6 ],
        ];
        
        $arrB = [
            [ 2, 2, 2 ],
            [ 2, 2, 2 ],
        ];
        
        $resArr = Math::hadamardProduct( $arrA, $arrB );
        $this->assertEquals( 
                [
                        [ 2, 4, 6 ],
                        [ 8, 10, 12 ],
                ], $resArr 
        );
        
        
        
    }
    
    
}


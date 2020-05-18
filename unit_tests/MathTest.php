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


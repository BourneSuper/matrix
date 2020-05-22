/*
 * Matrix is a PHP extension. It can do parallel computing base on CUDA.
 *
 * GitHub: https://github.com/BourneSuper/matrix
 *
 * Author: Bourne Wong <cb44606@gmail.com>
 *
 * */

#include <stdio.h>

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include "php.h"
#include "ext/standard/info.h"

#include <dev_util_p.h>


/**
 *
 */
void dup_HashTableTo1DArr( HashTable * hashTableP, double * arrP  ){
    int count = 0;
    zend_long hash;
    zend_string *key;
    zval *zvalue;
    zend_long h;
    zend_string *k;
    zval *zv;
    ZEND_HASH_FOREACH_KEY_VAL( hashTableP, hash, key, zvalue ){

        ZEND_HASH_FOREACH_KEY_VAL( Z_ARRVAL_P(zvalue), h, k, zv ){

            arrP[ count ] = zval_get_double_func(zv);

            count++;
        }ZEND_HASH_FOREACH_END();
    } ZEND_HASH_FOREACH_END();

}

/**
 *
 */
void dup_HashTableTo1DArrS( HashTable * hashTableP, float * arrP  ){
    int count = 0;
    zend_long hash;
    zend_string *key;
    zval *zvalue;
    zend_long h;
    zend_string *k;
    zval *zv;
    ZEND_HASH_FOREACH_KEY_VAL( hashTableP, hash, key, zvalue ){

        ZEND_HASH_FOREACH_KEY_VAL( Z_ARRVAL_P(zvalue), h, k, zv ){

            arrP[ count ] = (float)zval_get_double_func(zv);

            count++;
        }ZEND_HASH_FOREACH_END();
    } ZEND_HASH_FOREACH_END();

}

/**
 *
 */
void dup_HashTableTo1DArrOne( HashTable * hashTableP, double * arrP  ){
    int count = 0;
    zend_long hash;
    zend_string *key;
    zval *zvalue;
    ZEND_HASH_FOREACH_KEY_VAL( hashTableP, hash, key, zvalue ){
        arrP[ count ] = zval_get_double_func(zvalue);

        count++;
    } ZEND_HASH_FOREACH_END();

}

/**
 *
 */
void dup_HashTableTo1DArrOneS( HashTable * hashTableP, float * arrP  ){
    int count = 0;
    zend_long hash;
    zend_string *key;
    zval *zvalue;
    ZEND_HASH_FOREACH_KEY_VAL( hashTableP, hash, key, zvalue ){
        arrP[ count ] = (float)zval_get_double_func(zvalue);

        count++;
    } ZEND_HASH_FOREACH_END();

}


void dup_hashTableTo1DZval( HashTable * hashTableP, zval oneDimesionzval, int * shapeInfo, int * shapeInfoIndex ){
    zend_long hash;
    zend_string *key;
    zval * zvalue;
    int tempCount = 0;

    ZEND_HASH_FOREACH_KEY_VAL( hashTableP, hash, key, zvalue ){
        if( Z_TYPE_P( zvalue ) == IS_ARRAY ){
            ( * shapeInfoIndex )++;
            dup_hashTableTo1DZval( Z_ARRVAL_P(zvalue), oneDimesionzval, shapeInfo, shapeInfoIndex  );
            ( * shapeInfoIndex )--;
        }else{
            add_next_index_double( &oneDimesionzval, zval_get_double_func(zvalue) );
        }
        tempCount++;

    } ZEND_HASH_FOREACH_END();

    shapeInfo[ * shapeInfoIndex ] = tempCount;

}

void dup_oneDimesnPointerArrReshapeToZval( double * arrP, zval reshapedZval, int * shapeInfo, int * shapeInfoIndex, int * previousCount ){
    //
    int shapeInfoCount = 10;
    for( int i = 0; i < 10; i++ ){
        if( shapeInfo[i] == 0 ){
            shapeInfoCount = i;
            break;
        }
    }

    //
    if( shapeInfo[ * shapeInfoIndex ] == 0 ){
        return ;
    }

    if( * shapeInfoIndex == ( shapeInfoCount - 1 ) ){
        for( int i = 0; i < shapeInfo[ * shapeInfoIndex ]; i++ ){
            add_next_index_double( &reshapedZval, arrP[ ( * previousCount + i ) ] );
        }
        ( * previousCount ) += shapeInfo[ * shapeInfoIndex ];
    }else{
        for( int i = 0; i < shapeInfo[ * shapeInfoIndex ]; i++ ){
            zval tempZval;array_init( &tempZval );
            ( * shapeInfoIndex )++;
            dup_oneDimesnPointerArrReshapeToZval( arrP, tempZval, shapeInfo, shapeInfoIndex, previousCount );
            ( * shapeInfoIndex )--;
            add_next_index_zval( &reshapedZval, &tempZval );
        }
    }



}

void dup_oneDimensionZavlToPointerArr( zval * oneDimensionZavl, double * arrP ){
    HashTable * hashTableP = Z_ARRVAL_P(oneDimensionZavl);

    int count = 0;
    zend_long hash;
    zend_string *key;
    zval *zvalue;
    ZEND_HASH_FOREACH_KEY_VAL( hashTableP, hash, key, zvalue ){
        arrP[ count ] = (float)zval_get_double_func(zvalue);

        count++;
    } ZEND_HASH_FOREACH_END();
}



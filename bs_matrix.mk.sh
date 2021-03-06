#!/bin/sh

###########################################################################
#                                                                         #
# Matrix is a PHP extension. It can do parallel computing base on CUDA.   #
#                                                                         #
# GitHub: https://github.com/BourneSuper/matrix                           #
#                                                                         #
# Author: Bourne Wong <cb44606@gmail.com>                                 #
#                                                                         #
###########################################################################

echo "bs_matrix.mk.sh  running...";

##
insertLineNum=$((`cat  Makefile | grep -n "^all: " |awk -F: '{print $1}'` - 1));

`cat Makefile | head -n $insertLineNum > temp`;
`cat bs_matrix.mk.p1 >> temp `;
`cat Makefile | tail -n +$(($insertLineNum + 1)) >> temp`;

## 
`cat bs_matrix.mk.p2  >> temp`;

##
`cat temp > Makefile`;

echo "bs_matrix.mk.sh  running... done";
echo "Please use 'make bs_matrix' to make and install";

#!/bin/bash

for dir in ./*/
do
    cd $dir
    ./download.sh
    cd ..
done

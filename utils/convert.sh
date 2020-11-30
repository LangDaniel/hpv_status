#!/bin/bash

dir=$(dirname $1)
file=$(basename $1)
echo dir: $dir
echo file: $file

sudo docker run\
    -it\
    -v $PWD:/jobs\
    -w /jobs\
    -v $dir:/data\
    caffe_weights2hdf5 python convert.py /data/$file

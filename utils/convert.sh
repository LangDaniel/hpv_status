#!/bin/bash

dir=$(dirname $1)
echo $dir

sudo docker run -it -v $dir:/jobs -w /jobs caffe python convert.py $1

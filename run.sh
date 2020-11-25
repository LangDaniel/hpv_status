#!/bin/bash

dir=$PWD

cpu="0-5"
gpu=""

# docker
img=tf2-cv
remove="true"
run="d"
name="hpv_status"

usage() { echo "usage: $0 [-n NAME] [-i DOCKER IMAGE] [-g GPU] [-c CPU CORES] [-r <it|d>] [-d REMOVE IMAGE]" 1>&2; exit 1; }


while [[ $# -gt 0 ]];
do
    var="$1"
    if [ ${var::1} == "-" ] || [ ${var::2} == "--" ]
    then
        opt=${var//-}
        shift;
        arg="$1"
        case "$opt" in 
            "r"|"run")
                run="$arg"
                ;;
            "rm")
                remove="$arg"
                ;;
            "g"|"gpu")
                gpu="$arg"
                ;;
            "c"|"cpu")
                cpu="$arg"
                ;;
            "n"|"name")
                name="$arg"
                ;;
            *)
                usage
        esac
    else
        script=$var
        shift;
    fi
done

echo "running with flags: -$run --rm=$remove"
if [[ ! -z "${gpu// }" ]]; then
    echo "on CPUs $cpu and GPU $gpu"
else
    echo "on CPUs $cpu"
fi
read -p "Press enter to continue"

sudo docker run\
    -$run\
    --rm=$remove\
    --cpuset-cpus=$cpu\
    --gpus device=$gpu\
    --name $name\
    -v $dir:"/jobs"\
    -w "/jobs"\
    $img python $script 

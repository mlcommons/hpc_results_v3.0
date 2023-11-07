#!/bin/bash -e

if [ $# -lt 4 ] ; then
    #echo usage: $0 mapfile-header x-size y-size z-size
    echo usage: $0 node-shape blk-shape
    echo each shape = x y z
    echo example $0 16 4 16 16 2 16
    exit 1
fi

#map_head=$1
sizex=$1 && shift
sizey=$1 && shift
sizez=$1 && shift
blkx=$1 && shift
blky=$1 && shift
blkz=$1 && shift
num_of_nodes=$(($sizex * $sizey * $sizez))
blk_shape="${blkx}x${blky}x${blkz}"
numx=$((${sizex} / ${blkx}))
numy=$((${sizey} / ${blky}))
numz=$((${sizez} / ${blkz}))
numxz=$((${numx} * ${numz}))
blkxyz=$((${blkx} * ${blky} * ${blkz}))
numxyz=$((${numx} * ${numy} * ${numz}))

map_dir="./tmp_mapfile2/$blkxyz"
rm -rf $map_dir
mkdir -p $map_dir

for i in $(seq 0 $(($numxyz - 1)) ); do
    offx=$(($i % $numx * $blkx))
    offy=$(($i / numxz * $blky))
    offz=$(($i % numxz / $numx * $blkz))
    idx=`printf %04d $i`
    python mapper.py ${blk_shape} $offx,$offy,$offz > ${map_dir}/map_${idx}.txt
    test $? -eq 0 || exit 1
done

cat ${map_dir}/* | head -n $num_of_nodes

#rm -rf $map_dir

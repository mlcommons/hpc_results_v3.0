#!/bin/bash

SRC_DIR=$1
DST_DIR=$2
TYPE=$3
FILENAME_FILE=$4
N_TAR=$5
Compress=$6

WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
WORLD_RANK=$OMPI_COMM_WORLD_RANK

case $Compress in
    gz | gzip) CompressOpt="-I pigz"
               CompressExt=".gz" ;;
         lz4 ) CompressOpt="-I lz4"
               CompressExt=".lz4" ;;
          xz ) CompressOpt="-I pixz"
               CompressExt=".xz" ;;
        none ) CompressOpt=""
               CompressExt="" ;;
            *) echo "Unsupported compression type $Compress"
               exit 0;;
esac

Exec(){
    echo "$@"
    "$@"
}

cd $SRC_DIR/$TYPE

NLINES=`wc -l $FILENAME_FILE | cut -d " " -f 1`
LOCAL_NLINES=$(( $NLINES / $N_TAR ))

# e.g. train/0/cosmo_train_1.tar

for i in `seq $WORLD_RANK $WORLD_SIZE $(( $N_TAR - 1 ))`; do
    PARENT_DIR_NUM=$(($i / 1000))
    CHILD_FILE_NUM=$(($i % 1000))
    PARENT_DIR=${DST_DIR}/${TYPE}/${PARENT_DIR_NUM}
    mkdir -p ${PARENT_DIR}

    COMP_FILES=$SGE_LOCALDIR/files_${i}

    BEGIN_LINE=$(( $LOCAL_NLINES * $i + 1 ))
    END_LINE=$(( $LOCAL_NLINES * ($i + 1) ))

    sed -n ${BEGIN_LINE},${END_LINE}p $FILENAME_FILE > $COMP_FILES

    Exec tar ${CompressOpt} -cf ${PARENT_DIR}/cosmo_${TYPE}_${i}.tar${CompressExt} -T $COMP_FILES
done



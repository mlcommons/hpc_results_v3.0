#!/bin/bash -ex

echo `date +%s` `date` -- ompi_bind.sh_start

SIZE=${NumNodes}
RANK=${PMIX_RANK}
echo `date` "#start hostname: " `hostname` "JOBID: "${PJM_SUBJOBID}

df -t lliofs > /dev/null 2>&1
rm -rf /tmp/${USER} >/dev/null 2>&1 :

export SCRIPT_DIR=${ShareDir}

export CPSCRIPT="cpdata_decomp8K.sh"

#-- OPT file(SRC_FILE) defined in BatchBase
export OPT_PATH=${ShareDir}/TF220-33

#-- setenv
module list

export LD_LIBRARY_PATH=/lib64:/usr/lib64:${OPT_PATH}/lib:${LD_LIBRARY_PATH}
export PATH=${OPT_PATH}/bin:${PATH}

JobID=${PJM_SUBJOBID}
CheckFilePrefix="opt_copy_finished_"
CheckFile=${CheckFilePrefix}${JobID}

Host=`hostname`
HostPost=${Host: -1}
rm -f ${ShareDir}/${CheckFilePrefix}*
rm -rf ${ShareDir}/TF220* :
rm -f ${ShareDir}/cpdata.sh :

time -p tar -I pigz -xf ${SRC_FILE} -C ${ShareDir}
cp ./${CPSCRIPT} ${ShareDir}/cpdata.sh

#TMP_DEST=${LocalDataDir}
TMP_DEST=/local/tmp
mkdir -p ${TMP_DEST}
chmod 777 -R ${TMP_DEST}
umask 000 ${TMP_DEST}

. "$ParameterFile"

LogDir=${LOGDIR}

export TF_MKL_OPTIMIZE_PRIMITIVE_MEMUSE=false
export TMPDIR=$TMP_DEST #for python tempfile

ulimit -s 16384

if [ $RANK -eq "0" ]; then
    export DNNL_VERBOSE=$dnnlverbose
    env > ${LogDir}/rank0_env
    ulimit -a > ${LogDir}/rank0_ulimit
fi

echo `date +%s` `date` -- cp opt DONE!

which numactl
which python

LIBTCMALLOC_DIR="${OPT_PATH}/lib"
export LD_PRELOAD=${LIBTCMALLOC_DIR}/libtcmalloc.so
PERF=""

time -p numactl --cpunodebind 4-7 --membind 4-7 \
    ${PERF} \
    python "train.py" "${PARAMS[@]}"

unset LD_PRELOAD

echo `date +%s` `date` -- python DONE!

if [ ${HostPost} = 'b' ];then
  sleep 10
  rm -rf ${ShareDir}
fi
rm -rf ${TMP_DEST}

echo `date +%s` `date` -- ALL DONE!

exit

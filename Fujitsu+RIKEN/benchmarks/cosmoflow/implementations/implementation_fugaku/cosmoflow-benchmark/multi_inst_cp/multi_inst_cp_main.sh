#!/bin/bash

SrcDir="$1" && shift
DstDir="$1" && shift
CompType="$1" && shift
TotalNumTar="$1" && shift
NumNodePerInst="$1" && shift
NumInst="$1" && shift
BeginFlag="$1" && shift
EndFlag="$1" && shift
CpProg="$1" && shift
UntarProg="$1" && shift

export OPT_PATH=${ShareDir}/TF220-33
export LD_LIBRARY_PATH=/lib64:/usr/lib64:${OPT_PATH}/lib:${LD_LIBRARY_PATH}
export PATH=${OPT_PATH}/bin:${PATH}

TmpDir=/worktmp

NumTarLocal=$(( $TotalNumTar / $NumNodePerInst ))

# trainとvalidationで2倍の個数のtarを展開する
$UntarProg $TmpDir $DstDir $CompType $(( $NumTarLocal * 2)) $EndFlag &
UntarPID=$!

$CpProg $SrcDir $DstDir $TmpDir $CompType $TotalNumTar $NumInst $BeginFlag $EndFlag

wait $UntarPID

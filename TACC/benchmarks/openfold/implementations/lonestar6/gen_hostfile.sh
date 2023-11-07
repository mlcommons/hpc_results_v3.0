#!/bin/bash

ppn=$1
if [[ $ppn == "" ]]
then
  echo -e $red"Correct usage: ./gen_hostsfile.sh <#processes on each node>"
  echo -e $red"Exiting..."
  exit 0
fi

if [ ! -z $SLURM_JOBID ]
then
    JOBID=$SLURM_JOBID
    hostfile="hosts.$JOBID"
    rm -f $hostfile
    for i in `scontrol show hostnames $SLURM_NODELIST`
    do
      for (( j=0; j<$ppn; j++ ))
      do
        echo $i>>$hostfile
      done
    done
fi

h="`wc -l $hostfile | cut -d \" \" -f1`"
if [ -s $hostfile ]
then
  echo "$hostfile file created with $h procs."
fi
cp hosts.$SLURM_JOBID hosts


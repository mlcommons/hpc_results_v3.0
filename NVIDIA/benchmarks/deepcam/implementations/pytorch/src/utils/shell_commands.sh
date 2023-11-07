#!/bin/bash

# The MIT License (MIT)
#
# Modifications Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

log_time() {
    local file=${1:?Must provide an output file with MLLOG tags}
    local start_time=$(cat ${file} | grep submission_benchmark | awk '{split($5,a,","); print a[1]}')
    local end_time=$(cat ${file} | grep run_stop | awk '{split($5,a,","); print a[1]}')
    local duration=$(echo $end_time $start_time | awk '{print ($1-$2)/1000./60.}')
    echo "Log time [m]: $duration"
}

run_time() {
  local file=${1:?Must provide an output file with MLLOG tags}
  local start_time=$(cat ${file} | grep run_start | awk '{split($5,a,","); print a[1]}')
  local end_time=$(cat ${file} | grep run_stop | awk '{split($5,a,","); print a[1]}')
  local duration=$(echo $end_time $start_time | awk '{print ($1-$2)/1000./60.}')
  echo "Run time [m]: $duration"
}

run_epochs() {
  local file=${1:?Must provide an output file with MLLOG tags}
  local end_epochs=$(cat ${file} | grep target_accuracy_reached | awk '{split($18,a,","); print a[1]}')
  local end_steps=$(cat ${file} | grep target_accuracy_reached | awk '{split($20,a,"}"); print a[1]}')
  echo "Number of epochs: ${end_epochs} (${end_steps} steps)"
}

init_time() {
  local file=${1:?Must provide an output file with MLLOG tags}
  local start_time=$(cat ${file} | grep init_start | awk '{split($5,a,","); print a[1]}')
  local end_time=$(cat ${file} | grep init_stop | awk '{split($5,a,","); print a[1]}')
  local duration=$(echo $end_time $start_time | awk '{print ($1-$2)/1000./60.}')
  echo "Initialization time [m]: $duration"
}

staging_time() {
  local file=${1:?Must provide an output file with MLLOG tags}
  local start_time=$(cat ${file} | grep staging_start | awk '{split($5,a,","); print a[1]}')
  local end_time=$(cat ${file} | grep staging_stop | awk '{split($5,a,","); print a[1]}')
  local duration=$(echo $end_time $start_time | awk '{print ($1-$2)/1000./60.}')
  echo "Staging time [m]: $duration"
}

eval_time() {
    local file=${1:?Must provide an output file with MLLOG tags}
    cat ${file} | grep eval_start | awk '{split($5,a,","); print a[1]}' > .tmp_start
    cat ${file} | grep eval_stop | awk '{split($5,a,","); print a[1]}' > .tmp_stop
    local durations=$(paste .tmp_stop .tmp_start | awk '{print ($1-$2)/1000.}')
    rm .tmp_start .tmp_stop
    local stats=$(echo $durations | awk '{min=10000.; max=0.; mean=0.; for(i=1; i<=NF; i++){mean+=$i/NF;
                                                                                      if(max < $i){ max = $i;};
                                                                                      if(min > $i){ min = $i}; }; print mean, min, max}')
    local mean=$(echo ${stats} | awk '{print $1}')
    local min=$(echo ${stats} | awk '{print $2}')
    local max=$(echo ${stats} | awk '{print $3}')
    echo "Epoch time [s]:"
    echo "    mean: ${mean}"
    echo "    min: ${min}"
    echo "    max: ${max}"
}

epoch_time() {
    local file=${1:?Must provide an output file with MLLOG tags}
    local drop_epochs=${2:-0}
    cat ${file} | grep epoch_start | awk '{split($5,a,","); print a[1]}' > .tmp_start
    cat ${file} | grep epoch_stop | awk '{split($5,a,","); print a[1]}' > .tmp_stop
    paste .tmp_stop .tmp_start > .tmp
    if [ ${drop_epochs} -ge 1 ]; then
	count=$(cat .tmp | wc -l)
	tail -n$(( ${count} - ${drop_epochs} )) .tmp > .tmp_out
	mv .tmp_out .tmp
    fi
    local durations=$(cat .tmp | awk '{print ($1-$2)/1000.}')
    rm .tmp_start .tmp_stop .tmp
    local stats=$(echo $durations | awk '{min=10000.; max=0.; mean=0.; for(i=1; i<=NF; i++){mean+=$i/NF;
                                                                                      if(max < $i){ max = $i;};
                                                                                      if(min > $i){ min = $i}; }; print mean, min, max}')
    local mean=$(echo ${stats} | awk '{print $1}')
    local min=$(echo ${stats} | awk '{print $2}')
    local max=$(echo ${stats} | awk '{print $3}')
    echo "Epoch time [s]:"
    echo "    mean: ${mean}"
    echo "    min: ${min}"
    echo "    max: ${max}"
}

storage_bandwidth() {
    # input
    local file=${1:?Must provide an output file with MLLOG tags}

    # constant
    local sample_size=$(( 1152 * 768 * (16 + 1) * 4 ))

    # get data from file
    cat ${file} | grep epoch_start | awk '{split($5,a,","); print a[1]}' > .tmp_start
    cat ${file} | grep epoch_stop | awk '{split($5,a,","); print a[1]}' > .tmp_stop
    local durations=$(paste .tmp_stop .tmp_start | awk '{print ($1-$2)/1000.}')
    rm .tmp_start .tmp_stop
    local stats=$(echo $durations | awk '{min=10000.; max=0.; mean=0.; for(i=1; i<=NF; i++){mean+=$i/NF;
                                                                                      if(max < $i){ max = $i;};
                                                                                      if(min > $i){ min = $i}; }; print mean, min, max}')

    # get samples
    local num_samples=$(cat ${file} | grep train_samples | awk '{split($11,a,","); print a[1]}')
    local batch_size=$(cat ${file} | grep global_batch_size | awk '{split($11,a,","); print a[1]}')
    local num_samples=$(( (${num_samples} / ${batch_size}) * ${batch_size} ))
    echo $num_samples, ${sample_size}
    local tot_size=$(( ${sample_size} * ${num_samples} ))

    # compute bandwidth
    local gb=$(( 1024 * 1024 * 1024 ))
    local min=$(echo ${stats} | awk -v ts=${tot_size} -v gb=${gb} '{print ts/($3 * gb)}')
    local max=$(echo ${stats} | awk -v ts=${tot_size} -v gb=${gb} '{print ts/($2 * gb)}')
    local mean=$(echo ${stats} | awk -v ts=${tot_size} -v gb=${gb} '{print ts/($1 * gb)}')

    echo "Storage Bandwidth [GB/s]:"
    echo "    mean: ${mean}"
    echo "    min: ${min}"
    echo "    max: ${max}"
}


step_time() {
    local file=${1:?Must provide an output file with MLLOG tags}

    # project the timings                                                                                                                                          
    cat ${file} | grep train_loss | awk '{split($5,a,","); split($20,b,"}"); print a[1],b[1]}' > .tmp
    local nstamps=$(cat .tmp | wc -l)
    cat .tmp | head -n $(( ${nstamps} - 1 )) > .tmpm
    cat .tmp | tail -n $(( ${nstamps} - 1 )) > .tmpp
    paste .tmpp .tmpm | awk '{print ($1-$3)/($2-$4)}' > .tmpd

    # compute stats                                                                                                                                                
    local stats=$(cat .tmpd | awk 'BEGIN{min=10000.; max=0.; mean=0.; n=0}{n+=1;
                                                                           mean+=$i;
                                                                           if(max < $i){ max = $i;};
                                                                           if(min > $i){ min = $i};}
                                                                           END{print mean/n, min, max}')

    local mean=$(echo ${stats} | awk '{print $1}')
    local min=$(echo ${stats} | awk '{print $2}')
    local max=$(echo ${stats} | awk '{print $3}')
    echo "Step time [ms]:"
    echo "    mean: ${mean}"
    echo "    min: ${min}"
    echo "    max: ${max}"
}



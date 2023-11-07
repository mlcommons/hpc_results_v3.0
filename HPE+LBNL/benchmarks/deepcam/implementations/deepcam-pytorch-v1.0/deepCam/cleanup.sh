#!/bin/bash


# weak scaling copy
for file in $(ls ${OUTPUT_DIR}/logs/*instance*.log); do
    instance_id=$(echo ${file} | awk '{split($1,a,"instance"); split(a[2], b, ".log"); print b[1]}')
    target="${LOGFILE_BASE}_${instance_id}_${EXP_ID}.log"
    if [ ! -f ${target} ]; then
	cp ${file} ${target}
    fi
done

# strong scaling copy
if [ -f ${OUTPUT_DIR}/logs/${RUN_TAG}_${EXP_ID}.log ]; then
    target="${LOGFILE_BASE}_${EXP_ID}.log"
    if [ ! -f ${target} ]; then
	cp ${OUTPUT_DIR}/logs/${RUN_TAG}_${EXP_ID}.log ${target}
    fi
fi

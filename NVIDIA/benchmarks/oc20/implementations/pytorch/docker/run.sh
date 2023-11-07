docker run --runtime=nvidia -it --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 --rm -v ${PWD}:/workspace/oc20 -v ${DATA}:/data --privileged --ipc=host oc20 /bin/bash

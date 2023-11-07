base_image=registry.nersc.gov/das/pytorch:23.09-py3
target_image=registry.nersc.gov/das/sfarrell/opencatalyst-opt:23.09.01
podman-hpc build --build-arg FROM_IMAGE_NAME=$base_image \
    -t $target_image -f Dockerfile .

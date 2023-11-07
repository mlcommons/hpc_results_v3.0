docker build --platform linux/amd64 \
    --build-arg FROM_IMAGE_NAME=nvcr.io/nvdlfwea/pytorch:23.09-py3 \
    -t sfarrell/opencatalyst-opt:23.09.00 -f Dockerfile .

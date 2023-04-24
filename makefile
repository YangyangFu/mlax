
HOST = yangyangfu

# define image names
JAX_VERSION=0.4.8
IMAGE_NAME = jax-cuda12

# build 
build:
	docker build --no-cache --rm -t ${HOST}/${IMAGE_NAME}:${JAX_VERSION} .

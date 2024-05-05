docker rm -f openpose

DOCKER_FOLDER=$(pwd)
REPO_FOLDER=$(dirname $DOCKER_FOLDER)

docker run --gpus all -it --network=host --name openpose -v $REPO_FOLDER:/workspace woodenheart/openpose bash

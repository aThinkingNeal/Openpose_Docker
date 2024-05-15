# remove old container
docker rm -f openpose_deploy

docker run --gpus all -it --network=host --name openpose_deploy openpose:final  bash
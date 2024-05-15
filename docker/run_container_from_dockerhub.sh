docker pull woodenheart777/openpose:deploy_0514

docker rm -f openpose_deploy

docker run --gpus all -it --network=host --name openpose_deploy woodenheart/openpose:0515 bash
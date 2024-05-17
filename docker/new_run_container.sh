# remove old container
docker rm -f woodenheart/openpose:0517

docker run --gpus all -p 5000:5000 woodenheart/openpose:0517
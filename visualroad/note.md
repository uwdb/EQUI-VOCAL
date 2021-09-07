# Run the Docker container
docker run --runtime=nvidia -v /home/ubuntu/complex_event_video/visualroad:/home/ue4/visualroad --rm -dti --ipc=host --name visualroad visualroad/core

docker run --runtime=nvidia --rm -dti --ipc=host --name ue4 adamrehn/ue4-engine:4.22.0

# Install PIL
pip3 install Pillow

# Kill process
sudo apt install net-tools
netstat -tulpn
kill [pid]
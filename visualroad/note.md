# Carla examples

```python
actor_list = world.get_actors()
```

- Print the location of all the speed limit signs in the world. 
```python
for speed_sign in actor_list.filter('traffic.speed_limit.*'):
    print(speed_sign.get_location())
```

docker run --runtime=nvidia -v /home/ubuntu/visualroad:/home/ue4/visualroad --rm -dti --ipc=host --name visualroad visualroad/core

docker run --runtime=nvidia --rm -dti --ipc=host --name ue4 adamrehn/ue4-engine:4.22.0

sudo apt install net-tools
netstat -tulpn
kill [pid]
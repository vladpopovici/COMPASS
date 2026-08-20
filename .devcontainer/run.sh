#!/usr/bin/bash

DVER=0.1
docker run -it --rm \
    --user $(id -u):$(id -g) \
    --volume /home/vlad/Projects:/Projects \
    --hostname devil \
    --name devil \
    --publish 127.0.0.1:8889:8888/tcp \
    compass-py3_13:${DVER} /bin/bash

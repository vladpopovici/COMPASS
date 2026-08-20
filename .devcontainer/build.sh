#!/usr/bin/bash

DVER=0.1

docker buildx build \
    --rm \
    --build-arg dock_uid=$( id -u ) \
    --build-arg dock_gid=$( id -g ) \
    --tag compass-py3_13  .

docker tag compass-py3_13:latest compass-py3_13:${DVER} 

#!/bin/bash

echo 'Building Dockerfile with image name pymarl'
nohup docker build --build-arg UID=$UID -t pymarl:ppo . & 

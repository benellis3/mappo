#!/usr/bin/env bash
python3 setup.py build_ext --inplace


#export PYTHONPATH=$PYTHONPATH:"/home/cs/Documents/deepmarl-coma/src"
#cython --embed  history.pyx -o history.c
#gcc -shared -fPIC -I/usr/include/python3.5 -o gabber.so history.c -lpython3.5m
#python3 setup.py build_ext --inplace
#cython --version
#cython --embed history.py -o history.c
#gcc $(pkg-config --libs --cflags python3) history.c -o history

#cython history.py
#cython replay_buffer.py
#cython runner.pyx
#cython scheme.pyx
#cython transforms.pyx
#g++ -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python3.5 -o blitzz.so history.c replay_buffer.c runner.c scheme.c transforms.c
#python3 setup.py build_ext -i
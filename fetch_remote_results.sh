#!/bin/bash
rsync -chavzP --stats ${1}@${2}:~/projects/pymarl/results ./

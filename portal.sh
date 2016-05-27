#!/bin/bash

rm main.cu
cp main.c main.cu
./client -u grupo28 -x fJe21qYN -q cuda main.cu 1024 1024 500 1 0 0
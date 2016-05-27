#!/bin/bash

rm main.cu
cp main.c main.cu
./client -u grupo28 -x fJe21qYN -q cuda main.cu 100 100 35 1 5 3

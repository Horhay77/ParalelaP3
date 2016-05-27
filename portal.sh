#!/bin/bash

rm main.cu
cp PendienteDeEnvio.c main.cu
./client -u grupo28 -x fJe21qYN -q cuda main.cu 512 512 100 1 0 0
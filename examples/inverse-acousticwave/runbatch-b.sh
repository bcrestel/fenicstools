#!/bin/bash

paramks='8e-6 7e-6 6e-6'

for paramk in $paramks
do
    echo paramk: $paramk

    echo param: b
    mpirun -n 30 python inversion-acousticwave.py b $paramk > output/k$paramk+b
done

echo Bash completed for param b

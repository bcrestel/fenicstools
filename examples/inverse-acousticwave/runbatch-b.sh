#!/bin/bash

for paramk in {6..2..1}
do
    echo paramk: $paramk

    echo param: b
    mpirun -n 30 python inversion-acousticwave.py b $paramk > output/k$paramk+b
done

echo Bash completed

#!/bin/bash

paramks='0.0 1e-12 1e-10 1e-8 1e-6 1e-4'

for paramk in $paramks
do
    echo paramk: $paramk

    echo param: ab
    mpirun -n 30 python inversion-acousticwave.py ab $paramk > output/k$paramk+ab
done

echo Bash completed param a

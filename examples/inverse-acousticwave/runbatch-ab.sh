#!/bin/bash

paramks='8e-6 7e-6 6e-6 9e-6 5e-6 1e-5 1e-6'

for paramk in $paramks
do
    echo paramk: $paramk

    echo param: ab
    mpirun -n 30 python inversion-acousticwave.py ab $paramk > output/k$paramk+ab
done

echo Bash completed param ab

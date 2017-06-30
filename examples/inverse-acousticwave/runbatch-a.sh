#!/bin/bash

paramks='8e-6 7e-6 6e-6 4e-6 2e-6'

for paramk in $paramks
do
    echo paramk: $paramk

    echo param: a
    mpirun -n 30 python inversion-acousticwave.py a $paramk > output/k$paramk+a
done

echo Bash completed param a

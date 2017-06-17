#!/bin/bash

for paramk in {6..2..1}
do
    echo paramk: $paramk

    echo param: a
    mpirun -n 30 python inversion-acousticwave.py a $paramk > output/k$paramk+a
done

echo Bash completed

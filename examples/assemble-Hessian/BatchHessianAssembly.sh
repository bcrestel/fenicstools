#!/bin/bash

# Run all serial runs to compute different parts of the data-misfit Hessian
# matrix

if [ $# -ne 1 ]; then
    echo $0: usage: need to provide number of processes
    exit 1
fi

NBPROC=$1
LASTPROC=`expr $NBPROC - 1`

for i in $(eval echo {0..$LASTPROC}); 
do 
    nohup python assemble-Hessian-b.py $i $NBPROC > 'assemble-Hessian-b-'$i'.out' &
done

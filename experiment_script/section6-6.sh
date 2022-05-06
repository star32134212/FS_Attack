#!/bin/bash
for i in `seq 1 100`;
do
    echo $i
    for m in "lrp"
    do
        python exp_comparision.py --method $m --n 500 --cuda --origin True --num $i --dump True --D True
    done
done

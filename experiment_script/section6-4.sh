#!/bin/bash
for i in `seq 1 100`;
do
    echo $i
    for m in "gradient" "guided_backprop" "grad_times_input" "integrated_grad" "lrp"
    do
        python exp_comparision.py --method $m --n 1000 --cuda --origin True --num $i --D True
    done
done

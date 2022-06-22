#!/bin/bash
for i in `seq 0 500`;
do
     python attack_effect_experiment.py --method "integrated_grad" --n 1000 --cuda --num $i --topk 95 --early_stop True --F "IG" --dump True
done

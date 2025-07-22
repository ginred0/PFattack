#!/bin/bash
for seed in {0..9}
do
    nohup python PFAttack.py  --seed $seed --training_rounds 2  --epochs 20 --communication_rounds 20 --select_rate 0.5 --infered_weight 1 --device 3  --attack_type s --attacker_rate 0.05 --dataset ACSPublicCoverage  --beta 1.5 --alpha 0.2 --gamma 10 --aggragation qfed  --defense  trimmed_median > result/PFAttack_ACSPublicCoverage_${seed}_qFed_trimmed_median.out  2>&1 &
done


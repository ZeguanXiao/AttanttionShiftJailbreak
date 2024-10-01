#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

python src/main.py --task-name "attacker-vicuna-target-llama-2-example-group1-1-10-5" \
--attack-model vicuna-13b --attack-max-n-tokens 1024 --attack-device "cuda:0" --attack-batch-size 1 \
--target-model llama-2  --target-max-n-tokens 2048  --target-device "cuda:1" --target-batch-size 2 \
--judge-model deberta --judge-model-device "cuda:1" --judge-model-batch-size 8  \
--n-global-iterations 1 --n-streams 10 --n-iterations 5 --example-group 1 \
--seed 2025

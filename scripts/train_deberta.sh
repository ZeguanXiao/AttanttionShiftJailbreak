#!/bin/bash

MODEL_NAME="microsoft/deberta-v3-large"
OUTPUT_DIR=PATH/TO/CLASSIFIER/OUTPUT_DIR

torchrun --nproc_per_node 1 finetune_deberta.py \
  --model_name_or_path ${MODEL_NAME} \
  --output_dir ${OUTPUT_DIR} \
  --train_file "data/deberta_data/train.csv" \
  --validation_file "data/deberta_data/evaluate.csv"

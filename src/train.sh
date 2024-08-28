#!/bin/bash

MODEL_LOCATION="esm2_t6_8M_UR50D"
FASTA_DIR="/nashome/uglee/training/data/data_ver2"
SAVE_DIR="~/"
OUTPUT_DIM=3
NUM_BLOCKS=5
BATCH_SIZE=16
LEARNING_RATE=0.0005
NUM_EPOCHS=20
USE_GPU=true

CMD="python3 train.py $MODEL_LOCATION $FASTA_DIR $SAVE_DIR $OUTPUT_DIM $NUM_BLOCKS $BATCH_SIZE $LEARNING_RATE $NUM_EPOCHS"

if [ "$USE_GPU" = false ]; then
    CMD="$CMD --nogpu"
fi

echo "Running command: $CMD"
$CMD

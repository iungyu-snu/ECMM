#!/bin/bash

MODEL_LOCATION="esm2_t6_8M_UR50D"
FASTA_DIR="/nashome/uglee/training/data/data_ver3"
SAVE_DIR="/nashome/uglee/ECMM/tests"
OUTPUT_DIM=2
NUM_BLOCKS=2
BATCH_SIZE=32
LEARNING_RATE=0.001
NUM_EPOCHS=25
USE_GPU=true
WEIGHT_DECAY=0

CMD="python3 train.py $MODEL_LOCATION $FASTA_DIR $SAVE_DIR $OUTPUT_DIM $NUM_BLOCKS $BATCH_SIZE $LEARNING_RATE $NUM_EPOCHS --weight_decay $WEIGHT_DECAY"

if [ "$USE_GPU" = false ]; then
    CMD="$CMD --nogpu"
fi

echo "Running command: $CMD"
$CMD

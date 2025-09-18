#!/bin/bash

INPUT_CSV="/home/lixiaojie/code/genview-clip/StableRep/data_generation/cc3m_50w-3rd/missing.csv"
OUTPUT_CSV="/home/lixiaojie/code/genview-clip/StableRep/data_generation/cc3m_50w-3rd/score_50w-3rd_part2.csv"
MODEL_NAME="deepseek"

# Multi-GPU
CUDA_DEVICES="0,1,2,3,4,5,6,7"
IFS=',' read -r -a GPU_LIST <<< "$CUDA_DEVICES"
NUM_GPUS=${#GPU_LIST[@]}  # 计算 GPU 数量

echo "Using GPUs: ${GPU_LIST[*]} for parallel processing."

for ((i=0; i<$NUM_GPUS; i++))
do
    GPU_ID=${GPU_LIST[i]}
    echo "Starting process on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python score.py \
        --input_csv "$INPUT_CSV" \
        --output_csv "$OUTPUT_CSV" \
        --model_name "$MODEL_NAME" \
        --gpu_rank $i \
        --n_gpus $NUM_GPUS &
done

wait
echo "All GPU processes finished!"


# Single-GPU:
# CUDA_VISIBLE_DEVICES=3 python score.py \
#         --input_csv "$INPUT_CSV" \
#         --output_csv "$OUTPUT_CSV" \
#         --model_name "$MODEL_NAME" \
#         --gpu_rank 0 --n_gpus 1

#!/bin/bash

# 1views
work_dir="/data3/datasets/wangbei_data/local/300w-50wSyn/real-txt-img-imgtxt-Ada_QD-20e-gamma2-0-0-0-1"

epochs=90
model="base"
pretrained="$work_dir/epoch_last.pth"

# enable dataset gpu_ids batch_size drop_last use_bn
TASK_CONFIGS=(
  "1  in1k        0,1,2,3,4,5,6,7 256   1   0"
  "1  cifar10     0,1             256   1   0"
  "1  in100       2,3             256   1   0"
  "1  cifar100    2,3             256   1   0"
  "1  dtd         2               64    0   0"
  "1  flowers     4               64    0   0"
  "1  food101     5,6             256   1   0"
  "1  sun397      4               256   1   0"
  "1  aircraft    7               128   0   0"
  "1  pets        7               128   0   0"
  "1  caltech101  1               128   0   0"
)


base_port=60000
index=0

for config in "${TASK_CONFIGS[@]}"; do
  read -r enable task gpus batch_size drop_last use_bn <<< "$config"

  if [ "$enable" -ne 1 ]; then
    echo "Skipping $task"
    ((index++))
    continue
  fi

  echo "Launching $task on GPUs $gpus"

  num_gpu=$(echo "$gpus" | awk -F',' '{print NF}')
  port=$((base_port + index))
  linear_dir="linear_${task}_${num_gpu}xb${batch_size}_${epochs}e"

  cmd="CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node=$num_gpu \
    --nnodes=1 --node_rank=0 --master_port=$port \
    main_linear.py --model $model --epochs $epochs \
    --pretrained $pretrained \
    --output-dir $work_dir/$linear_dir \
    --dataset $task \
    --batch-size $batch_size"

  if [ "$drop_last" -eq 1 ]; then
    cmd="$cmd --drop_last"
  fi
  if [ "$use_bn" -eq 1 ]; then
    cmd="$cmd --use_bn"
  fi

  eval "$cmd &"

  ((index++))
done

wait
# Multi-GPU
node_idx=0
n_nodes=1
gpu_list=(4 5 6 7)
n_gpus=${#gpu_list[@]}

for ((i=0;i<$n_gpus;i++)); do
    export CUDA_VISIBLE_DEVICES=${gpu_list[i]}
    python data_generation/generate.py --outdir data/CC3M/gen_v2/ \
    --conditioned_mode 'imgtxt' --scale 6.0 --batch_size 1 \
    --from_file data/CC3M/test_gen.csv \
    --root_path /data1/datasets/CC3M/raw \
    --gpu_idx $i --n_gpus $n_gpus --node_idx $node_idx --n_nodes $n_nodes \
    --n_samples 1 --img_save_size 256 --seed 0 &
done

# Single-GPU
# node_idx=0
# n_nodes=1
# n_gpus=1
# export CUDA_VISIBLE_DEVICES=0

# python data_generation/generate.py --outdir data/CC3M/gen/ \
# --conditioned_mode 'img' --scale 2.0 --batch_size 1 \
# --from_file data/CC3M/test_gen.csv  \
# --root_path /data1/datasets/CC3M/raw \
# --noise_level 0 \
# --gpu_idx 0 --n_gpus $n_gpus --node_idx $node_idx --n_nodes $n_nodes \
# --n_samples 1 --img_save_size 256


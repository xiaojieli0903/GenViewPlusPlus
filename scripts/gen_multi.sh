node_idx=0          # index of this computing node (useful for multi-node training)
n_nodes=1           # total number of nodes
gpu_list=(0 1 2 3)  # list of GPUs to use
n_gpus=${#gpu_list[@]}  # number of GPUs

for ((i=0;i<$n_gpus;i++)); do
    export CUDA_VISIBLE_DEVICES=${gpu_list[i]}
    python data_generation/generate.py \
    --outdir data/output \
    --conditioned_mode 'imgtxt' \
    --guidance_scale 10.0 \
    --noise_level 100 \
    --batch_size 1 \
    --from_file data/CC3M/infos.csv \
    --root_path /path/to/raw/images \
    --gpu_idx $i \
    --n_gpus $n_gpus \
    --node_idx $node_idx \
    --n_nodes $n_nodes \
    --n_samples 1 \
    --img_save_size 256 &
done
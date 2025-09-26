node_idx=0
n_nodes=1
n_gpus=1
export CUDA_VISIBLE_DEVICES=0

python data_generation/generate.py \
    --outdir data/output \
    --conditioned_mode 'imgtxt' \
    --guidance_scale 10.0 \
    --noise_level 100 \
    --batch_size 1 \
    --from_file data/CC3M/infos.csv \
    --root_path /path/to/raw/images \
    --gpu_idx 0 \
    --n_gpus $n_gpus \
    --node_idx $node_idx \
    --n_nodes $n_nodes \
    --n_samples 1 \
    --img_save_size 256
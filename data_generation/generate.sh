# Multi-GPU
node_idx=0
n_nodes=1
gpu_list=(0 1 2 3 4 5 6 7)
n_gpus=${#gpu_list[@]}

for ((i=0;i<$n_gpus;i++)); do
    export CUDA_VISIBLE_DEVICES=${gpu_list[i]}
    python txt2img_ours.py --outdir /data1/datasets/CC3M/txt2img-Ada/score2 \
    --conditioned_mode 'txt' --scale 6.0 --batch_size 1 \
    --from_file /home/lixiaojie/code/genview-clip/StableRep/data_generation/score_10w/score2.csv \
    --root_path /data1/datasets/CC3M/raw \
    --gpu_idx $i --n_gpus $n_gpus --node_idx $node_idx --n_nodes $n_nodes \
    --n_samples 1 --img_save_size 512 &
done

# Single-GPU
node_idx=0
n_nodes=1
n_gpus=1
export CUDA_VISIBLE_DEVICES=0

python txt2img_ours_v2.py --outdir /data3/datasets/CC3M/0408test \
--conditioned_mode 'imgtxt' --scale 2.0 --batch_size 1 \
--from_file /data3/datasets/CC3M/0408test/csv/add_noise0.csv  \
--root_path /data1/datasets/CC3M/raw \
--noise_level 0 \
--gpu_idx 0 --n_gpus $n_gpus --node_idx $node_idx --n_nodes $n_nodes \
--n_samples 2 --img_save_size 512


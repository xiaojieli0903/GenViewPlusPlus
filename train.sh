torchrun --nproc_per_node=4 --nnodes=1 \
  --node_rank=0 --master_port=20001 \
  main_stablerep.py \
    --model base \
    --batch_size 64 \
    --epochs 25 --warmup_epochs 2 \
    --blr 2.0e-4 --weight_decay 0.1 --beta1 0.9 --beta2 0.98 \
    --num_workers 8 \
    --output_dir ./output/100w-50wSyn/real-txt-img-imgtxt-Ada_QD-16e-gamma2-1-1-1-1_25e \
    --log_dir ./output/100w-50wSyn/real-txt-img-imgtxt-Ada_QD-16e-gamma2-1-1-1-1_25e \
    --csv_path /data1/datasets/CC3M/cc3m_100w_relative.csv \
    --folder_list /data1/datasets/CC3M/raw \
                  /data3/datasets/CC3M/txt_scale-ada_noise100_times1 \
                  /data3/datasets/CC3M/img_scale10.0_noise-ada_times1 \
                  /data3/datasets/CC3M/imgtxt_scale-ada_noise-ada_times1 \
    --folder_suffix_list .jpg .jpg .jpg .jpg \
    --real_images_path_suffix /data1/datasets/CC3M/raw .jpg \
    --n_img 4 --downsample --downsample_prob 0.05 --down_res 64 128 \
    --syn_idx_list 1 2 3 --syn_ratio 1.0 \
    --syn_csv_path /data1/datasets/CC3M/cc3m_50w_relative.csv \
    --gamma 2 --epoch_switch 16 \
    --early_loss_coefs 1 0 1 0 --later_loss_coefs 1 1 1 1 \

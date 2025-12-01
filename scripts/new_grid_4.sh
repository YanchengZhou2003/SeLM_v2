cnt=0
ratio_dyn=0.8

for N in 2048 4096 8192 16384 32768 65536 131072; do
    for tp in 15 9 3; do
        for N_dynbr in 128 256 512 1024 2048; do
            if [ $N_dynbr -eq 128 ]; then
                block_size=1024
            elif [ $N_dynbr -eq 256 ]; then
                block_size=1024
            elif [ $N_dynbr -eq 512 ]; then
                block_size=512
            elif [ $N_dynbr -eq 1024 ]; then
                block_size=256
            elif [ $N_dynbr -eq 2048 ]; then
                block_size=128
            fi
            N_top=$(($N_dynbr / 2))

            echo "Running experiment with N=$N, tp=$tp, N_dynbr=$N_dynbr, ratio_dyn=$ratio_dyn"
            python -m src.gpt  \
                --N_train $N    --T_train $block_size  \
                --N_vocab 65    --T_vocab 65  \
                --N_valid 2048  --T_valid $block_size  \
                --N_dynbr $N_dynbr   --N_stnbr 65   \
                --N_top $N_top    --pos_ratio 0.875 \
                --ratio_dyn $ratio_dyn --step_dyn 1 \
                --h 13 --tp $tp --c 0.8 --cur_tp 2 --cur_portion 1.00  \
                --train_epoch_num 80 --valid_epoch_num 50 \
                --save_interval   40 \
                --train_converge 0 --valid_converge 0 \
                --train_graph_reset 1 --valid_graph_reset 1 \
                --train_only 0 --valid_only 0 \
                --val_interval 10 --vis_interval 80 \
                --use_eu_norm 0 --temperature 10 \
                --vis_path ./vis/vis_ICML_4/N${N}_tp${tp}_Ndynbr${N_dynbr}_ratio${ratio_dyn} \
                --use_filter 0 \
                --train_save_path ./ckpt/cte/N${N}_tp${tp}_Ndynbr${N_dynbr}_ratio${ratio_dyn}.pt \
                > ./logs/log_ICML_4/N${N}_tp${tp}_Ndynbr${N_dynbr}_ratio${ratio_dyn}.log 2>&1
            
            wait

            echo "Experiment with N=$N, tp=$tp, N_dynbr=$N_dynbr, ratio_dyn=$ratio_dyn completed."
            done
        done
    done
done



for N in 1024 2048 4096 8192 16384; do
    for tp in 3 6 9 12 15; do
        for pos_ratio in 0.125 0.25 0.5 0.75 0.875; do
            for ratio_dyn in 0.80 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.90 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99; do
                block_size=512
                echo "Running experiment with N=$N, tp=$tp, temperature=$t, ratio_dyn=$ratio_dyn, block_size=$block_size"
                python -m src.gpt  \
                    --N_train $N    --T_train $block_size  \
                    --N_vocab 65    --T_vocab 65  \
                    --N_valid 2048  --T_valid $block_size  \
                    --N_dynbr 512   --N_stnbr 65   \
                    --N_top 256     --pos_ratio $pos_ratio \
                    --ratio_dyn $ratio_dyn --step_dyn 1 \
                    --h 13 --tp $tp --c 1.0 --cur_tp 2 --cur_portion 1.00  \
                    --train_epoch_num 100 --valid_epoch_num 100 \
                    --train_converge 0 --valid_converge 0 \
                    --train_graph_reset 1 --valid_graph_reset 1 \
                    --train_only 0 --valid_only 0 \
                    --val_interval 10 --vis_interval 50 \
                    --use_eu_norm 0 --temperature 10 \
                    --vis_path ./vis/vis_ICML_322/N${N}_tp${tp}_temp${t}_ratio${ratio_dyn}_pos${pos_ratio}/ \
                    --use_filter 0 \
                    --train_save_path ./ckpt/cte/N${N}_tp${tp}_temp${t}_ratio${ratio_dyn}_pos${pos_ratio}.pt \
                    > ./logs/log_ICML_3/N${N}_tp${tp}_temp${t}_ratio${ratio_dyn}_pos${pos_ratio}.log 2>&1
                wait
                echo "Experiment with N=$N, tp=$tp, pos_ratio=$pos_ratio, ratio_dyn=$ratio_dyn completed."
            done
        done
    done
done
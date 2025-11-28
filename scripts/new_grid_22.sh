for N in 1024 2048 4096; do
    for tp in 7 5 3; do
        for t in 10; do
            for ratio_dyn in 0.86 0.88 0.90 0.92 0.94 0.96 ; do
                block_size=256
                echo "Running experiment with N=$N, tp=$tp, temperature=$t, ratio_dyn=$ratio_dyn, block_size=$block_size"
                python -m src.gpt  \
                    --N_train $N  --T_train $block_size  \
                    --N_vocab 65    --T_vocab 65  \
                    --N_valid 512   --T_valid $block_size  \
                    --N_dynbr 512  --N_stnbr 65   \
                    --N_top 256   \
                    --ratio_dyn $ratio_dyn --step_dyn 1 \
                    --h 13 --tp $tp --c 1.0 --cur_tp 2 --cur_portion 1.00  \
                    --train_epoch_num 100 --valid_epoch_num 100 \
                    --train_converge 0 --valid_converge 0 \
                    --train_graph_reset 25 --valid_graph_reset 25 \
                    --train_only 0 --valid_only 0 \
                    --val_interval 10 --vis_interval 50 \
                    --use_eu_norm 0 --temperature $t \
                    --vis_path ./vis/vis_ICML_22/N${N}_tp${tp}_temp${t}_ratio${ratio_dyn}/ \
                    --use_filter 0 \
                    --train_save_path ./ckpt/cte/N${N}_tp${tp}_temp${t}_ratio${ratio_dyn}.pt \
                    > ./logs/log_ICML_22/N${N}_tp${tp}_temp${t}_ratio${ratio_dyn}.log 2>&1
                wait
                echo "Experiment with N=$N, tp=$tp, temperature=$t, ratio_dyn=$ratio_dyn completed."
            done
        done
    done
done
for N in 16384; do
    for tp in 4; do
        for t in 10; do
            for pos_ratio in 0.5; do
                for ratio_dyn in 0.96 ; do
                    block_size=512
                    echo "Running experiment with N=$N, tp=$tp, temperature=$t, ratio_dyn=$ratio_dyn, block_size=$block_size"
                    python -m src.gpt  \
                        --N_train $N    --T_train $block_size  \
                        --N_vocab 65    --T_vocab 65  \
                        --N_valid 2048  --T_valid $block_size  \
                        --N_dynbr 512  --N_stnbr 65   \
                        --ratio_dyn $ratio_dyn --step_dyn 1 \
                        --h 13 --tp $tp --c 1.0 --cur_tp 2 --cur_portion 1.00  \
                        --train_epoch_num 50 --valid_epoch_num 50 \
                        --train_converge 0 --valid_converge 0 \
                        --train_graph_reset 1 --valid_graph_reset 1 \
                        --train_only 0 --valid_only 0 \
                        --val_interval 10 --vis_interval 50 \
                        --use_eu_norm 0 --temperature $t \
                        --vis_path ./vis/vis_tmp/N${N}_tp${tp}_temp${t}_ratio${ratio_dyn}/ \
                        --use_filter 0 \
                        --train_save_path ./ckpt/cte/N${N}_tp${tp}_temp${t}_ratio${ratio_dyn}.pt \
                        > ./logs/log_tmp/tmp.log 2>&1
                    wait
                    
                done
            done
        done
    done
done
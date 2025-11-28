cnt=0

for N in 8192 16384; do
    for tp in 3 9 15; do
        for pos_ratio in 0.125 0.5 0.875; do
            for ratio_dyn in 0.80 0.82 0.84 0.86 0.88 0.90 0.92 0.94 0.96 0.98; do
                block_size=512
                echo "Running experiment with N=$N, tp=$tp, pos_ratio=$pos_ratio, ratio_dyn=$ratio_dyn"
                python -m src.gpt  \
                    --N_train $N    --T_train $block_size  \
                    --N_vocab 65    --T_vocab 65  \
                    --N_valid 2048  --T_valid $block_size  \
                    --N_dynbr 512   --N_stnbr 65   \
                    --N_top 256     --pos_ratio $pos_ratio \
                    --ratio_dyn $ratio_dyn --step_dyn 1 \
                    --h 13 --tp $tp --c 0.8 --cur_tp 2 --cur_portion 1.00  \
                    --train_epoch_num 100 --valid_epoch_num 100 \
                    --train_converge 0 --valid_converge 0 \
                    --train_graph_reset 1 --valid_graph_reset 1 \
                    --train_only 0 --valid_only 0 \
                    --val_interval 10 --vis_interval 50 \
                    --use_eu_norm 0 --temperature 10 \
                    --vis_path ./vis/vis_ICML_322/N${N}_tp${tp}_ratio${ratio_dyn}_pos${pos_ratio}/ \
                    --use_filter 0 \
                    --train_save_path ./ckpt/cte/N${N}_tp${tp}_ratio${ratio_dyn}_pos${pos_ratio}.pt \
                    > ./logs/log_ICML3/N${N}_tp${tp}_ratio${ratio_dyn}_pos${pos_ratio}.log 2>&1 &
                
                cnt=$((cnt+1))
                if [ $cnt -eq $max_cnt ]; then
                    cnt=0
                    wait
                fi

                echo "Experiment with N=$N, tp=$tp, pos_ratio=$pos_ratio, ratio_dyn=$ratio_dyn completed."
            done
        done
    done
done
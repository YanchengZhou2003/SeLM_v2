cnt=0

for N in 8192; do
    for tp in 16; do
        block_size=512
        echo "Running experiment with N=$N, tp=$tp"

        python -m src.gpt  \
            --N_train $N    --T_train $block_size  \
            --N_vocab 65    --T_vocab 65  \
            --N_valid 2048  --T_valid $block_size  \
            --N_dynbr 512   --N_stnbr 65   \
            --N_top 256     --pos_ratio 0.5 \
            --ratio_dyn 0.5 --step_dyn 1 \
            --h 13 --tp $tp --c 1.0 --cur_tp 2 --cur_portion 1.00  \
            --train_epoch_num 40 --valid_epoch_num 100 \
            --train_converge 0 --valid_converge 0 \
            --train_graph_reset 1 --valid_graph_reset 1 \
            --train_only 1 --valid_only 0 \
            --val_interval 100 --vis_interval 100 \
            --use_eu_norm 0 --temperature 10 \
            --vis_path ./vis/tmp/ \
            --use_filter 0 \
            --train_save_path ./ckpt/cte/tmp.pt \
            # > ./logs/speed_baseline/N${N}_tp${tp}_pipelined.log 2>&1 

        wait

        echo "Experiment with N=$N, tp=$tp, completed."
        
    done
done

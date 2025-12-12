cnt=0
pos_ratio=0.875

for N in 8192; do
    for ratio_dyn in 0.95; do
        for tp in 3; do
            block_size=256
            echo "Running experiment with N=$N, tp=$tp, pos_ratio=$pos_ratio, ratio_dyn=$ratio_dyn"
            python -m src.gpt  \
                --N_train $N    --T_train $block_size  \
                --N_vocab 65    --T_vocab 65  \
                --N_valid 2048  --T_valid $block_size  \
                --N_dynbr 512  --N_top   256   \
                --N_dynbr_v 512 --N_top_v   256   \
                --N_stnbr 65    --pos_ratio $pos_ratio \
                --ratio_dyn $ratio_dyn  --step_dyn 1 \
                --h 13 --tp $tp --c 0.8 --cur_tp 3 --cur_portion 0.75 \
                --train_epoch_num 100    --valid_epoch_num 50          \
                --train_converge 0      --valid_converge 0            \
                --train_graph_reset 1   --valid_graph_reset 1         \
                --train_only 0   --valid_only   0 \
                --val_interval 1 --vis_interval 20 \
                --use_eu_norm 0  --temperature 10 --gt_temperature 1  \
                --save_interval 100 \
                --vis_path    ./vis/20251129_phase_transition/N${N}_tp${tp}_ratio${ratio_dyn}_pos${pos_ratio}/ \
                --use_filter  0 \
                --train_save_path ./ckpt/cte/tmp.pt \
                > ./logs/20251129_phase_transition/exp1/N${N}_tp${tp}_ratio${ratio_dyn}_pos${pos_ratio}.log 2>&1 &
            
            wait

            echo "Experiment with N=$N, tp=$tp, pos_ratio=$pos_ratio, ratio_dyn=$ratio_dyn completed."
        done
    done
done
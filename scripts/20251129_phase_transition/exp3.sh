cnt=0
pos_ratio=0.875

for N in 8192; do
    for h in 13 10; do
        for tp in 1 2 4 9; do
            for ratio_dyn in 0.00 0.0001 0.001 0.01 0.02 0.04 0.08 0.16 0.32 0.64 0.80 0.99 1.00; do
                for N_dynbr in 4 8 16 32 64 128 256 512; do
                    for N_dynbr_v in 4 8 16 32 64 128 256 512; do
                        N_top=$((N_dynbr / 2))
                        N_top_v=$((N_dynbr_v / 2))
                        T_train=$((512 * 512 / N_dynbr))
                        T_valid=$((512 * 512 / N_dynbr_v))

                        T_train=$(( N < T_train ? N : T_train ))
                        T_valid=$(( 2048 < T_valid ? 2048 : T_valid ))

                        echo "Running experiment with N=$N, tp=$tp, pos_ratio=$pos_ratio, ratio_dyn=$ratio_dyn"
                        python -m src.gpt  \
                            --N_train $N    --T_train $T_train  \
                            --N_vocab 65    --T_vocab 65  \
                            --N_valid 2048  --T_valid $T_valid  \
                            --N_dynbr $N_dynbr  --N_top   $N_top   \
                            --N_dynbr_v $N_dynbr_v --N_top_v $N_top_v   \
                            --N_stnbr 65    --pos_ratio $pos_ratio \
                            --ratio_dyn $ratio_dyn  --step_dyn 1 \
                            --h $h --tp $tp --c 0.8 --cur_tp 3 --cur_portion 0.75 \
                            --train_epoch_num 60    --valid_epoch_num 60          \
                            --train_converge 0      --valid_converge 0            \
                            --train_graph_reset 1   --valid_graph_reset 1         \
                            --train_only 0   --valid_only   0 \
                            --val_interval 1 --vis_interval 60 \
                            --use_eu_norm 0  --temperature 10 --gt_temperature 1  \
                            --save_interval 60 \
                            --vis_path    ./vis/20251129_phase_transition/exp3/ratio${ratio_dyn}_t${N_dynbr}_v${N_dynbr_v}_h${h}_tp${tp} \
                            --use_filter  0 \
                            --train_save_path ./ckpt/cte/ratio_dyn${ratio_dyn}_t${N_dynbr}_h${h}_tp${tp}.ckpt \
                            > ./logs/20251129_phase_transition/exp3/ratio${ratio_dyn}_t${N_dynbr}_v${N_dynbr_v}_h${h}_tp${tp}.log 2>&1 &
                        
                        wait

                        echo "Experiment with N=$N, tp=$tp, pos_ratio=$pos_ratio, ratio_dyn=$ratio_dyn completed."
                    done                        
                done
            done
        done
    done
done

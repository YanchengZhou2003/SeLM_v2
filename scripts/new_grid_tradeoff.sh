sleep 1800

for h in 11; do
    for tp in 8; do
        for N in 8192 16384 32768 131072; do
            for N_vocab in 512; do
                for pos_ratio in 0.125 0.5 0.875; do
                    for ratio_dyn in 0.00 0.01 0.04 0.16 0.64 0.90 0.95 0.99 1.00; do
                        for ratio_sta in 0.00 0.01 0.04 0.16 0.64 0.90 0.95 0.99 1.00; do
                            if (( $(echo "$ratio_dyn + $ratio_sta > 1.0" | bc -l) )); then
                                echo "Skipping invalid configuration: ratio_dyn=$ratio_dyn, ratio_sta=$ratio_sta"
                                continue
                            fi

                            log="./logs/log_tradeoff2/h${h}_tp${tp}_N${N}_vocab${N_vocab}_pos${pos_ratio}_dyn${ratio_dyn}_sta${ratio_sta}.log"
                            if [ -f "$log" ]; then
                                echo "Skipping existing log: $log"
                                continue
                            fi

                            T_train=512
                            T_valid=512

                            echo "Running experiment with h=$h, tp=$tp, N=$N, N_vocab=$N_vocab, pos_ratio=$pos_ratio, ratio_dyn=$ratio_dyn, ratio_sta=$ratio_sta"
                            python -m src.gpt  \
                                --N_train $N    --T_train $T_train  \
                                --N_vocab $N_vocab    --T_vocab $N_vocab  \
                                --N_valid 2048  --T_valid $T_valid  \
                                --N_dynbr 512  --N_top   256   \
                                --N_dynbr_v 512 --N_top_v 256   \
                                --N_stnbr $N_vocab    --pos_ratio $pos_ratio \
                                --ratio_dyn $ratio_dyn  --ratio_sta $ratio_sta --step_dyn 1 \
                                --h $h --tp $tp --c 0.8 --cur_tp 3 --cur_portion 0.75 \
                                --train_epoch_num 60    --valid_epoch_num 60          \
                                --train_converge 0      --valid_converge 0            \
                                --train_graph_reset 1   --valid_graph_reset 1         \
                                --train_only 0   --valid_only   0 \
                                --val_interval 10 --vis_interval 20 \
                                --use_eu_norm 0  --temperature 10 --gt_temperature 1  \
                                --save_interval 60 \
                                --vis_path    ./vis/vis_tradeoff2/h${h}_tp${tp}_N${N}_vocab${N_vocab}_pos${pos_ratio}_dyn${ratio_dyn}_sta${ratio_sta} \
                                --use_filter  0 \
                                --train_save_path ./ckpt/cte/h${h}_tp${tp}_N${N}_vocab${N_vocab}_pos${pos_ratio}_dyn${ratio_dyn}_sta${ratio_sta}2.ckpt \
                                > ./logs/log_tradeoff2/h${h}_tp${tp}_N${N}_vocab${N_vocab}_pos${pos_ratio}_dyn${ratio_dyn}_sta${ratio_sta}.log 2>&1 &
                            
                            wait

                            echo "Experiment with h=$h, tp=$tp, N=$N, N_vocab=$N_vocab, pos_ratio=$pos_ratio, ratio_dyn=$ratio_dyn, ratio_sta=$ratio_sta completed."
                        done
                    done
                done
            done
        done
    done
done
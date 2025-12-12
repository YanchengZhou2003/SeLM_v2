
for N in 2048 8192 32768; do
    for N_vocab in 384 512 768; do
        for pos_ratio in 0.125 0.5 0.875; do
            for ratio_dyn in 0.00 0.01 0.02 0.04 0.08 0.16 0.32 0.64 0.80 0.95 0.99 1.00; do
                T_train=512
                T_valid=256

                echo "Running experiment with N=$N, N_vocab=$N_vocab, pos_ratio=$pos_ratio, ratio_dyn=$ratio_dyn"
                python -m src.gpt  \
                    --N_train $N    --T_train $T_train  \
                    --N_vocab $N_vocab    --T_vocab $N_vocab  \
                    --N_valid 2048  --T_valid 2048  \
                    --N_dynbr 512  --N_top   256   \
                    --N_dynbr_v 512 --N_top_v 256   \
                    --N_stnbr $N_vocab    --pos_ratio $pos_ratio \
                    --ratio_dyn $ratio_dyn  --step_dyn 1 \
                    --h 11 --tp 4 --c 0.8 --cur_tp 3 --cur_portion 0.75 \
                    --train_epoch_num 60    --valid_epoch_num 60          \
                    --train_converge 0      --valid_converge 0            \
                    --train_graph_reset 1   --valid_graph_reset 1         \
                    --train_only 0   --valid_only   0 \
                    --val_interval 1 --vis_interval 20 \
                    --use_eu_norm 0  --temperature 10 --gt_temperature 1  \
                    --save_interval 60 \
                    --vis_path    ./vis/vocabup/N${N}_vocab${N_vocab}_ps${pos_ratio}_ratio${ratio_dyn} \
                    --use_filter  0 \
                    --train_save_path ./ckpt/cte/N${N}_vocab${N_vocab}_ps${pos_ratio}_ratio${ratio_dyn}.ckpt \
                    > ./logs/log_vocabup/N${N}_vocab${N_vocab}_ps${pos_ratio}_ratio${ratio_dyn}.log 2>&1 &
                
                wait

                echo "Experiment with N=$N, N_vocab=$N_vocab, pos_ratio=$pos_ratio, ratio_dyn=$ratio_dyn completed."
            done
        done
    done
done

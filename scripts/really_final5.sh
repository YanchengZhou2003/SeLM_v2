h=18
tp=16
step_dyn=1

for N in 65536 131072; do
    for ratio_dyn in 0.00 0.05 0.10 0.20 0.40 0.80 1.00; do 
        python -m src.gpt  \
            --N_train $N    --T_train 16  \
            --N_vocab 65    --T_vtnbr 64  \
            --N_valid 512   --T_valid 16  \
            --N_dynbr 256   --N_stnbr 65 \
            --ratio_dyn $ratio_dyn --step_dyn $step_dyn \
            --h $h --tp $tp --cur_tp 1 --cur_portion 1.0 --division_fact 1.0 \
            --train_epoch_num 200 --valid_epoch_num 200 \
            --train_converge 0 --valid_converge 0 \
            --train_graph_reset 20 --valid_graph_reset 20 \
            --train_only 0 --valid_only 0 \
            --val_interval 10 --vis_interval 50 \
            --use_eu_norm 0 --temperature 10.0 \
            --vis_path ./vis/vis_final6/N${N}_rd${ratio_dyn} \
            --use_filter 0 \
            --train_save_path ./ckpt/cte/N${N}_rd${ratio_dyn}.pt \
            > ./logs/log_final6/N${N}_rd${ratio_dyn}.log 2>&1 &
    done
    wait
    done
done
valid_epoch_num=200
h=18
tp=6

for N in 131072; do
    for step_dyn in 1 2 4; do
        for ratio_dyn in 0.00 0.05 0.10 0.15 0.20 0.30 0.40 0.50; do 
            train_epoch_num=$((100 * step_dyn))
            python -m src.gpt  \
                --N_train $N    --T_train 32  \
                --N_vocab 65    --T_vtnbr 128  \
                --N_valid 512   --T_valid 32  \
                --N_dynbr 256   --N_stnbr 65 \
                --ratio_dyn $ratio_dyn --step_dyn $step_dyn \
                --h $h --tp $tp --cur_tp 3 --cur_portion 0.60 --division_fact 1.0 \
                --train_epoch_num $train_epoch_num --valid_epoch_num $valid_epoch_num \
                --train_converge 0 --valid_converge 0 \
                --train_graph_reset 15 --valid_graph_reset 15 \
                --train_only 0 --valid_only 0 \
                --val_interval 10 --vis_interval 50 \
                --use_eu_norm 0 --temperature 10.0 \
                --vis_path ./vis/vis_final5/N${N}_rd${ratio_dyn}_sd${step_dyn} \
                --use_filter 0 \
                --train_save_path ./ckpt/cte/N${N}_rd${ratio_dyn}_sd${step_dyn}.pt \
                > ./logs/log_final5/N${N}_rd${ratio_dyn}_sd${step_dyn}.log 2>&1 &
        done
        wait
    done
done
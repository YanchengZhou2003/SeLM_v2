h=6
N=16384
N_vocab=65
pos_ratio=0.875
epoch_num=200

N_dynbr=512
N_dynbr_v=512
ratio_dyn=0.5
ratio_sta=$(awk -v r="$ratio_dyn" 'BEGIN{printf "%.5f", 1.0 - r}')

for tp in 36; do
    vis_path=./vis/vis_tp/tp_${tp}
    save_path=./ckpt/cte/cte_tp_${tp}.pt

    T_valid=$((512 * 512 / N_dynbr_v))
    T_train=$((256 * 512 / N_dynbr))
    N_top_v=$((N_dynbr_v / 2))
    N_top=$((N_dynbr / 2))

    echo "Running experiment with tp=$tp"

    python -m src.gpt  \
        --N_train $N            --T_train $T_train  \
        --N_vocab $N_vocab      --T_vocab $N_vocab  \
        --N_valid 16384         --T_valid $T_valid  \
        --N_dynbr   $N_dynbr    --N_top   $N_top   \
        --N_dynbr_v $N_dynbr_v  --N_top_v $N_top_v   \
        --N_stnbr   $N_vocab    --pos_ratio $pos_ratio \
        --ratio_dyn $ratio_dyn  --ratio_sta $ratio_sta --step_dyn 1 \
        --h $h --tp $tp --c 1 --cur_tp 2 --cur_portion 0.75\
        --train_epoch_num $epoch_num    --valid_epoch_num $epoch_num          \
        --train_converge 0      --valid_converge 0            \
        --train_graph_reset 1   --valid_graph_reset 1         \
        --train_only 0          --valid_only   0 \
        --val_interval 1        --vis_interval 50 \
        --use_eu_norm 0         --temperature  10 --gt_temperature 1  \
        --save_interval $epoch_num \
        --vis_path    $vis_path \
        --use_filter  0 \
        --train_save_path $save_path \
        > ./logs/log_tp/tp_${tp}.log 2>&1

    echo "Completed experiment with tp=$tp"
done



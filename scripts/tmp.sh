h=11
tp=6
N=16384
N_vocab=65
pos_ratio=1.0
epoch_num=100


for ratio_dyn in 0.998; do
    for N_dynbr in 1024; do
        rm -rf ./vis/tmp/
        rm ./ckpt/cte/tmp_${N_vocab}_epoch${epoch_num}.pt
        mkdir -p ./vis/tmp/

        for N_dynbr_v in 512; do    

            ratio_sta=$(awk -v r="$ratio_dyn" 'BEGIN{printf "%.5f", 1.0 - r}')
            T_valid=$((512 * 512 / N_dynbr_v))
            T_train=$((128 * 512 / N_dynbr))
            N_top_v=$((N_dynbr_v / 2))
            N_top=$((N_dynbr / 2))

            echo "ratio_sta=$ratio_sta"

            echo "Running experiment with h=$h, tp=$tp, N=$N, N_vocab=$N_vocab, pos_ratio=$pos_ratio, ratio_dyn=$ratio_dyn, ratio_sta=$ratio_sta"

            python -m src.gpt  \
                --N_train $N            --T_train $T_train  \
                --N_vocab $N_vocab      --T_vocab $N_vocab  \
                --N_valid 16384         --T_valid $T_valid  \
                --N_dynbr   $N_dynbr    --N_top   $N_top   \
                --N_dynbr_v $N_dynbr_v  --N_top_v $N_top_v   \
                --N_stnbr   $N_vocab    --pos_ratio $pos_ratio \
                --ratio_dyn $ratio_dyn  --ratio_sta $ratio_sta --step_dyn 1 \
                --h $h --tp $tp --c 1 --cur_tp 2 --cur_portion 0.75\
                --train_epoch_num $epoch_num    --valid_epoch_num 80          \
                --train_converge 0      --valid_converge 0            \
                --train_graph_reset 1   --valid_graph_reset 1         \
                --train_only 0          --valid_only   0 \
                --val_interval 20       --vis_interval 20 \
                --use_eu_norm 0         --temperature 10 --gt_temperature 1  \
                --save_interval $epoch_num \
                --vis_path    ./vis/tmp_${N_vocab}/ \
                --use_filter  0 \
                --train_save_path ./ckpt/cte/tmp_${N_vocab}.pt 

            echo "Experiment with h=$h, tp=$tp, N=$N, N_vocab=$N_vocab, pos_ratio=$pos_ratio, ratio_dyn=$ratio_dyn, ratio_sta=$ratio_sta completed."
        done
    done
done



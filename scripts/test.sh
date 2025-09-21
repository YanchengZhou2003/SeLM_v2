h=8
cnt=0
for N in 131072; do
    for train_ratio_cos in 0.90; do
        cnt=$((cnt + 1))
        train_ratio_cro=$(echo "1.0 - $train_ratio_cos" | bc)
        python -m src.gpt  \
            --N_train $N --T_train 512 --N_ttnbr 512 --K_vocab 65 \
            --N_vocab 65   --T_vocab 65  --N_vvnbr 65   \
            --N_valid 512 --T_valid 64 --N_vanbr 512 \
            --h $h --tp 4 --cur_tp 2 --cur_portion 1.0 --division_fact 1.0 \
            --train_epoch_num 10 --valid_epoch_num 10 --train_ratio_cos $train_ratio_cos --train_ratio_cro $train_ratio_cro \
            --train_converge 0 --valid_converge 0 \
            --train_graph_reset 5 --vocab_graph_reset 5 --valid_graph_reset 5 \
            --train_only 0 --valid_only 0 \
            --val_interval 10 --vis_interval 50 \
            --use_eu_norm 0 --temperature 0.1 \
            --vis_path ./vis_new/h${h}_N${N}_train_ratio_cos${train_ratio_cos}_cro${train_ratio_cro}/ # \
            # > log_new/h${h}_N${N}_train_ratio_cos${train_ratio_cos}_cro${train_ratio_cro}.log 2>&1 &
        # if [ $((cnt % 4)) -eq 0 ]; then
        #     wait
        # fi
    done
done

# 0.27
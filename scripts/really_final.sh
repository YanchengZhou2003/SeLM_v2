### epoch_num 改回来了吗
### tp 改回来了吗
for tp in 1 2; do
    for h in 8 13 18 23; do
        #####
        if [ $tp -eq 1 ]; then
            cur_tp=1
        else
            cur_tp=2
        fi
        #####
        if [ $tp -eq 1 ]; then
            block_size=128
        elif [ $tp -eq 2 ]; then
            block_size=128
        elif [ $tp -eq 4 ]; then
            block_size=64
        elif [ $tp -eq 8 ]; then
            block_size=64
        elif [ $tp -eq 16 ]; then
            block_size=32
        fi
        ####
        python -m src.gpt  \
            --N_train 131072 --T_train $block_size --N_ttnbr 512 --K_vocab 65 \
            --N_vocab 65     --T_vocab 65  --N_vvnbr 65   \
            --N_valid 8192   --T_valid $block_size --N_vanbr 512 \
            --h $h --tp $tp --cur_tp $cur_tp --cur_portion 1.0 --division_fact 1.0 \
            --train_epoch_num 100 --valid_epoch_num 150 --train_ratio_cos 0.90 --train_ratio_cro 0.10 \
            --train_converge 0 --valid_converge 0 \
            --train_graph_reset 5 --vocab_graph_reset 5 --valid_graph_reset 5 \
            --train_only 0 --valid_only 0 \
            --val_interval 10 --vis_interval 50 \
            --use_eu_norm 0 --temperature 0.1 \
            --vis_path ./vis_final/h${h}_tp${tp} \
            > log_final/h${h}_tp${tp}.log 2>&1 &

    done
done

wait
# 0.27
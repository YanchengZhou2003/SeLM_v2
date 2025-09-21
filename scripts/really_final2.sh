### epoch_num 改回来了吗
### tp 改回来了吗
block_size=128

for ratio_cos in 0.999 0.99 0.95 0.90 0.75 0.50 0.25; do
    for temperature in 0.1 0.5 0.25 0.125 0.0625 0.03125; do

        ratio_cro=$(echo "1.0 - $ratio_cos" | bc)

        python -m src.gpt  \
            --N_train 131072 --T_train $block_size --N_ttnbr 512 --K_vocab 65 \
            --N_vocab 65     --T_vocab 65  --N_vvnbr 65   \
            --N_valid 8192   --T_valid $block_size --N_vanbr 512 \
            --h 18 --tp 4 --cur_tp 4 --cur_portion 1.0 --division_fact 1.0 \
            --train_epoch_num 100 --valid_epoch_num 150 --train_ratio_cos $ratio_cos --train_ratio_cro $ratio_cro \
            --train_converge 0 --valid_converge 0 \
            --train_graph_reset 5 --vocab_graph_reset 5 --valid_graph_reset 5 \
            --train_only 0 --valid_only 0 \
            --val_interval 10 --vis_interval 50 \
            --use_eu_norm 0 --temperature 0.1 \
            --vis_path ./vis_final/ratio_cos${ratio_cos}_temp${temperature} \
            > log_final/ratio_cos${ratio_cos}_temp${temperature}.log 2>&1 &  
    done
    wait
done


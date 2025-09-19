for N_train in 8192 65536; do
    for h in 8 24; do
        if [ $h -eq 24 ]; then
            tps=(4 1)
        else
            tps=(32 1)
        fi
        for tp in ${tps[@]}; do
            for align in 512 16; do
                echo "================ N_train = $N_train ================="
                block_size=256
                train_ratio_cos=0.99
                train_ratio_cro=0.01
                if [ $h -eq 8 ] && [ $tp -eq 1 ]; then
                    train_ratio_cos=0.8
                    train_ratio_cro=0.2
                fi

                python -m src.gpt  \
                    --N_train $N_train --T_train $block_size --N_ttnbr 512 --N_tvnbr $align --K_vocab 64 \
                    --N_vocab 8192 --T_vocab $block_size --N_vtnbr $align  --N_vvnbr 512  \
                    --N_valid 8192 --T_valid $block_size  --N_vanbr 512 \
                    --h $h --tp $tp --cur_tp 4 --cur_portion 0.5 --division_fact 1.0 \
                    --train_epoch_num 150 --valid_epoch_num 100 --train_ratio_cos $train_ratio_cos --train_ratio_cro $train_ratio_cro \
                    --train_converge 20 --valid_converge 20 \
                    --train_graph_reset 10 --vocab_graph_reset 10 --valid_graph_reset 10 \
                    --train_only 0 --valid_only 0 \
                    --val_interval 10 --vis_interval 10 \
                    --use_eu_norm 0 --vis_path ./vis_new/test_1_h${h}_tp${tp}_align${align}_N_train${N_train}/ > ./log_new/test_1_h${h}_tp${tp}_align${align}_N_train${N_train}.log 2>&1
                wait
            done
        done
    done
done



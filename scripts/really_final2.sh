for N in 1024 2048 4096 8192 16384 32768; do
    for h in 12 16 20; do
        if [ $h -eq 20 ]; then
            block_size=32
        elif [ $h -eq 16 ]; then
            block_size=64
        else
            block_size=128
        fi

        for tp in 16 8 4 2; do
            if [ $tp -eq 2 ]; then
                cur_tp=2
            else 
                cur_tp=4
            fi


            python -m src.gpt  \
                --N_train $N    --T_train $block_size --N_ttnbr 512 \
                --N_vocab 65    --T_vtnbr 256  \
                --N_valid 8192  --T_valid $block_size --N_vanbr 512 \
                --h $h --tp $tp --cur_tp $cur_tp --cur_portion 0.8 --division_fact 1.0 \
                --train_epoch_num 150 --valid_epoch_num 150 \
                --train_converge 0 --valid_converge 0 \
                --train_graph_reset 10 --valid_graph_reset 10 \
                --train_only 0 --valid_only 0 \
                --val_interval 10 --vis_interval 25 \
                --use_eu_norm 0 --temperature 10.0 \
                --vis_path ./vis_final2/N${N}_h${h}_tp${tp} \
                > ./log_final2/N${N}_h${h}_tp${tp}.log 2>&1 &

        done

        wait

    done
done
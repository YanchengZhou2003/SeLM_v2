for h in 14 18 22; do
    for tp in 4 2 8 16 32; do
        if [ $tp -gt 4 ]; then
            cur_tp=4
        else
            cur_tp=2
        fi
        
        if [ $h -eq 14 ] && [ $tp -eq 4 ]; then
            continue
        elif [ $h -eq 14 ] && [ $tp -eq 2 ]; then
            continue
        elif [ $h -eq 14 ] && [ $tp -eq 8 ]; then
            continue
        fi

        python -m src.gpt --train_length 131072 --valid_length 4096 --ratio_cos 0.99992 --ratio_cro 0.00008 --sample_factor 2.0 --h $h --tp $tp --cur_tp $cur_tp --cur_portion 0.5 --epoch_num 1000 --valid_only_epoch 250 --converge 100 --N_T 64 --vis_path "vis3/binary_test_h_${h}_tp_${tp}" > logs/binary_test_h_${h}_tp_${tp}.log 2>&1 &
        wait
    done
done

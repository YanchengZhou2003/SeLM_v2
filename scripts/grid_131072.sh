for h in 22 13 16 19; do
    for tp in 12 8; do
        python -m src.gpt --train_length 65536 --truncate_valid 4096 --ratio_cos 0.993 --ratio_cro 0.007 --sample_factor 2.0 --N_T 256 --vis_path ./vis2/new_grid_131072_h${h}_tp${tp} \
        --h $h --tp $tp --cur_tp 2 \
        --epoch_num 1000 --converge 20 > logs/new_grid_131072_h${h}_tp${tp}.log 2>&1 &
        wait
    done
done
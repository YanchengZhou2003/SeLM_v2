for h in 12 14 16 18 20 22 4 8; do
    for tp in 1 2 3 4 5 6 7; do
        python -m src.gpt --train_length 131072 --truncate_valid 2048 --ratio_cos 0.992 --ratio_cro 0.008 --sample_factor 1.0 --instant_writeback 1 --N_T 512 --vis_path ./vis2/grid_131072_h${h}_tp${tp} \
        --h $h --tp $tp \
        --epoch_num 50 --converge 15 > logs/grid_131072_h${h}_tp${tp}.log 2>&1 &
    done
    wait
done
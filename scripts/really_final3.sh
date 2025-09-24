train_epoch_num=100
valid_epoch_num=100

for N in 1024 2048 4096 8192 16384 32768 65536 131072; do
    for N_dynbr in 325 65 130 195 260; do 
        for h in 14 18 22; do
            for tp in 2 4 8; do
                if [ $tp -eq 2 ]; then
                    cur_tp=2
                else 
                    cur_tp=4
                fi

                python -m src.gpt  \
                    --N_train $N    --T_train 64  \
                    --N_vocab 65    --T_vtnbr 128  \
                    --N_valid 8192  --T_valid 64  \
                    --N_dynbr $N_dynbr   --N_stnbr 65 \
                    --h $h --tp $tp --cur_tp $cur_tp --cur_portion 0.8 --division_fact 1.0 \
                    --train_epoch_num $train_epoch_num --valid_epoch_num $valid_epoch_num \
                    --train_converge 0 --valid_converge 0 \
                    --train_graph_reset 10 --valid_graph_reset 10 \
                    --train_only 0 --valid_only 0 \
                    --val_interval 10 --vis_interval 25 \
                    --use_eu_norm 0 --temperature 10.0 \
                    --vis_path ./vis/vis_final3/N${N}_Ndynbr${N_dynbr}_h${h}_tp${tp} \
                    --use_filter 0 \
                    > ./logs/log_final3/N${N}_Ndynbr${N_dynbr}_h${h}_tp${tp}.log 2>&1 &

            done
        done
        wait
    done
done
# N_valid=2048
# for N_vocab in 384 512 768 1024; do
#     for N_train in 1024 2048 4096 8192 16384 32768 65536 131072; do
#         echo "================ N_train = $N_train ================="
#         for pos_ratio in 0.125 0.5 0.875; do
#             echo "---------------- pos_ratio = $pos_ratio ----------------"
#             python -m src.gpt --N_train $N_train --N_valid $N_valid --pos_ratio $pos_ratio --N_vocab $N_vocab --N_stnbr $N_vocab
#         done
#     done
# done



N_valid=16384
for N_vocab in 65; do
    for N_train in 16384; do
        echo "================ N_train = $N_train ================="
        for pos_ratio in 1.0; do
            echo "---------------- pos_ratio = $pos_ratio ----------------"
            python -m src.gpt --N_train $N_train --N_valid $N_valid --pos_ratio $pos_ratio --N_vocab $N_vocab --N_stnbr $N_vocab
        done
    done
done
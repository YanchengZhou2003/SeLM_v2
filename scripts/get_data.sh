N_valid=2048
for N_train in 1024 2048 4096 8192 16384; do
    echo "================ N_train = $N_train ================="
    for pos_ratio in 0.125 0.25 0.5 0.75 0.875; do
        echo "---------------- pos_ratio = $pos_ratio ----------------"
        python -m src.gpt --N_train $N_train --N_valid $N_valid --pos_ratio $pos_ratio
    done
done
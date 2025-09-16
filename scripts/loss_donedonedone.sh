for N_train in 512 1024 2048 4096 8192 16384 32768 65536 131072 524288; do
  echo "================ N_train = $N_train ================="
  python -m src.gpt --train_ratio_cos 0.99985 --train_ratio_cro 0.00015 --N_train $N_train --N_valid 8192 --vis_path ./vis3/N_train${N_train}_but_valid_only/ > ./logs/N_train${N_train}_but_valid_only.log 2>&1
  wait
done
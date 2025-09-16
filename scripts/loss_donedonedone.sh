for N_train in 512 1024 2048 4096 8192 16384 32768 65536 131072 524288; do
  echo "================ N_train = $N_train ================="
  python -m src.gpt --train_ratio_cos 0.99985 --train_ratio_cro 0.00015 --N_train $N_train --N_valid 8192 --vis_path ./vis3/N_train${N_train}_but_valid_only/ > ./logs/N_train${N_train}_but_valid_only.log 2>&1
  wait
done


N_train=4096
for h in 14 16 18 20 22; do
  for tp in 4 8 16 24 32; do
    echo "================ h = $h, tp = $tp ================="
    if [ $h -eq 14 ] && [ $tp -eq 4 ]; then
      continue
    fi

    if [ $h -eq 14 ] && [ $tp -eq 8 ]; then
      continue
    fi

    python -m src.gpt --train_ratio_cos 0.99985 --train_ratio_cro 0.00015 --N_train $N_train --N_valid 8192 --h $h --tp $tp --cur_tp 4 --vis_path ./vis/cte_h${h}_tp${tp} --train_epoch_num 500 --valid_epoch_num 300 --train_converge 50 --valid_converge 50 > logs/cte_h${h}_tp${tp}.log 2>&1 &
    wait
  done
done
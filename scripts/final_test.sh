
for train_length in 256 512 1024 2048 4096 8192 16384 32768 65536 131072; do
  echo "================ train_length = $train_length ================="
  python -m src.gpt --train_length $train_length --truncate_valid $train_length --ratio_cos 0.9999 --ratio_cro 0.0001 --sample_factor 4.0 --h 21 --tp 64 --cur_tp 4 --cur_portion 0.5 --epoch_num 1000 --converge 100
done
python -m src.gpt --train_length 131072 --truncate_valid 4096 --ratio_cos 0.9999 --ratio_cro 0.0001 --sample_factor 2.0 --h 18 --tp 32 --cur_tp 4 --cur_portion 0.5 --epoch_num 1000 --converge 50 --N_T 128
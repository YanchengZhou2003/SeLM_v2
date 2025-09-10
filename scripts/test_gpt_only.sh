for train_length in 512 1024 2048 4096 8192 16384 131072 262144; do
  echo "================ train_length = $train_length ================="
  python -m src.gpt --train_length $train_length
done
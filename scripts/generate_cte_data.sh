cte_train_bs=512
cte_eval_bs=512
# cte_eval_iters=16
cte_eval_iters=1

# for cte_train_bs in 1 2 4 8 16 32 64 128; do
for cte_train_iters in 1; do
  echo "================ cte_train_iters = $cte_train_iters ================="
  python -m src.gpt --cte_train_bs $cte_train_bs --cte_train_iters $cte_train_iters --cte_eval_bs $cte_eval_bs --cte_eval_iters $cte_eval_iters
done
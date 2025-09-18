for N_train in 4096 8192 16384; do
  echo "================ N_train = $N_train ================="
python -m src.gpt  \
    --N_train 8192 --T_train 128 --N_ttnbr 1024 --N_tvnbr 1024 --K_vocab 128 \
    --N_vocab 8192 --T_vocab 128 --N_vtnbr 1024  --N_vvnbr 1024  \
    --N_valid 8192 --T_valid 128  --N_vanbr 1024 \
    --h 25 --tp 1 --cur_tp 1 --cur_portion 0.5 --division_fact 1.0 \
    --train_epoch_num 150 --valid_epoch_num 150 --train_ratio_cos 0.99 --train_ratio_cro 0.01 \
    --train_converge 20 --valid_converge 20 \
    --train_graph_reset 10 --vocab_graph_reset 10 --valid_graph_reset 10 \
    --train_only 0 --valid_only 0 \
    --val_interval 10 --vis_interval 10 \
    --use_eu_norm 0 --vis_path ./vis_new/test_1/


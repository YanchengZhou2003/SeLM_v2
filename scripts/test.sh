# 要把 zeros 弄走，要把 vocab loss 放回去


python -m src.gpt  \
    --N_train 4096    --T_train 256  \
    --N_vocab 65    --T_vtnbr 512  \
    --N_valid 512   --T_valid 256  \
    --N_dynbr 256   --N_stnbr 65 \
    --ratio_dyn 0.0 --step_dyn 1 \
    --h 10 --tp 1 --cur_tp 1 --cur_portion 1.00 --division_fact 1.0 \
    --train_epoch_num 200 --valid_epoch_num 200 \
    --train_converge 0 --valid_converge 0 \
    --train_graph_reset 15 --valid_graph_reset 15 \
    --train_only 0 --valid_only 0 \
    --val_interval 10 --vis_interval 50 \
    --use_eu_norm 0 --temperature 10.0 \
    --use_filter 0 \
    --train_save_path ./ckpt/cte/N${N}_rd${ratio_dyn}_sd${step_dyn}.pt
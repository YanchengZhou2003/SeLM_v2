python -m src.gpt  \
    --N_train 8192  --T_train 128  \
    --N_vocab 65    --T_vtnbr 256  \
    --N_valid 8192  --T_valid 128  \
    --N_dynbr 256   --N_stnbr 65   \
    --ratio_dyn 0.5 --step_dyn 3 \
    --h 18 --tp 4 --c 1.0 --cur_tp 1 --cur_portion 1.0  \
    --train_epoch_num 200 --valid_epoch_num 200 \
    --train_converge 0 --valid_converge 0 \
    --train_graph_reset 20 --valid_graph_reset 20 \
    --train_only 0 --valid_only 0 \
    --val_interval 10 --vis_interval 50 \
    --use_eu_norm 0 --temperature 10.0 \
    --vis_path ./vis/example/ \
    --use_filter 0 \
    --train_save_path ./ckpt/cte/example.pt \
    > ./logs/example.log

# 参数说明：
# N_train: 训练集大小；T_train: 训练集分块大小
# N_vocab: 词表大小；  T_vtnbr (T_vocabulary_train_neighbor)： static token 看向 dynamic token 时的分块大小
# N_dynbr (N_dynamic neighbor)：每个 dynamic token 可见的 dynamic 邻居数量
# N_stnbr (N_static  neighbor)：每个 static  token 可见的 static  邻居数量
# ratio_dyn: dynamic-dynamic 与 dynamic-static 的 loss 加权
# step_dyn : 每过 step_dyn 个 epoch 就更新一次 dynamic embeddings 
# h, tp, c: 高度, 维数, 和 connection 采样系数
# cur_tp, cur_portion: 每次更新 locations，不更新全部，只随机更新 tp 中的 cur_tp 个，更新百分之 cur_portion 个
# train_epoch_num, valid_epoch_num: 顾名思义
# train_converge, valid_converge: converge 之前都只看向 1 个邻居
# train_graph_reset, valid_graph_reset: 每过指定 epoch 就重新随机一次邻居
# train_only, valid_only: 是否仅训练 / 仅评估
# use_eu_norm: 是否使用欧式空间的 norm。【先别使用这个】
# use_filter: 是否使用欧式空间的 filter。【先别使用这个】
# temperature: 顾名思义
# vis_path: 可视化保存的路径
# train_save_path: cte 训练好后的保存路径
# > ./logs/example: 程序输出到的路径


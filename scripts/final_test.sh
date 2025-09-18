
# for train_length in 262144 524288; do
#     echo "================ train_length = $train_length ================="
#     python -m src.gpt --train_length $train_length --truncate_valid 4096 --ratio_cos 0.99992 --ratio_cro 0.00008 --sample_factor 2.0 --h 18 --tp 32 --cur_tp 4 --cur_portion 0.5 --epoch_num 500 --converge 100 --N_T 128 --vis_path "vis3/final_test_train_length_$train_length" > logs/final_test_train_length_$train_length.log 2>&1 &
#     wait
# done

for train_length in 512 1024 2048 4096 8192 16384 32768; do
    echo "================ train_length = $train_length ================="
    
    python -m src.gpt --train_length $train_length --valid_length 4096 --ratio_cos 0.99992 --ratio_cro 0.00008 --sample_factor 2.0 --h 18 --tp 4 --cur_tp 2 --cur_portion 0.5 --epoch_num 1000 --valid_only_epoch 300 --converge 100 --N_T 512 --vis_path "vis3/final_test_train_length_${train_length}_argmax_filtered_match" > logs/final_test_train_length_${train_length}_argmax_filtered_match.log 2>&1 &
    
    wait
done

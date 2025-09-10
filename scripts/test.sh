python -m src.gpt --train_length 1048576 --truncate_valid 8192 --ratio_cos 0.993 --ratio_cro 0.007 --sample_factor 0.4 --h 27 > logs/test_h_27.log 2>&1 &
python -m src.gpt --train_length 1048576 --truncate_valid 8192 --ratio_cos 0.993 --ratio_cro 0.007 --sample_factor 0.4 --h 23 > logs/test_h_23.log 2>&1 &
wait
python -m src.gpt --train_length 1048576 --truncate_valid 8192 --ratio_cos 0.993 --ratio_cro 0.007 --sample_factor 0.4 --h 20 > logs/test_h_20.log 2>&1 &
python -m src.gpt --train_length 1048576 --truncate_valid 8192 --ratio_cos 0.993 --ratio_cro 0.007 --sample_factor 0.4 --h 18 > logs/test_h_18.log 2>&1 &
wait
python -m src.gpt --train_length 1048576 --truncate_valid 8192 --ratio_cos 0.993 --ratio_cro 0.007 --sample_factor 0.4 --h 16 > logs/test_h_16.log 2>&1 &
python -m src.gpt --train_length 1048576 --truncate_valid 8192 --ratio_cos 0.993 --ratio_cro 0.007 --sample_factor 0.4 --h 14 > logs/test_h_14.log 2>&1 &
wait
python -m src.gpt --train_length 1024 --ratio_dyn_prob 1.00 --truncate_valid 256 --sample_factor 0.4
python -m src.gpt --train_length 2048 --ratio_dyn_prob 1.00 --truncate_valid 256 --sample_factor 0.4
python -m src.gpt --train_length 4096 --ratio_dyn_prob 1.00 --truncate_valid 256 --sample_factor 0.4

python -m src.gpt --train_length 1024 --ratio_dyn_prob 1.00 --truncate_valid 256 --sample_factor 1.0
python -m src.gpt --train_length 2048 --ratio_dyn_prob 1.00 --truncate_valid 256 --sample_factor 1.0
python -m src.gpt --train_length 4096 --ratio_dyn_prob 1.00 --truncate_valid 256 --sample_factor 1.0

python -m src.gpt --train_length 1024 --ratio_dyn_prob 1.00 --truncate_valid 256 --sample_factor 2.5
python -m src.gpt --train_length 2048 --ratio_dyn_prob 1.00 --truncate_valid 256 --sample_factor 2.5
python -m src.gpt --train_length 4096 --ratio_dyn_prob 1.00 --truncate_valid 256 --sample_factor 2.5
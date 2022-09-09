gpu_ids=4
export CUDA_VISIBLE_DEVICES=${gpu_ids}

log_dir="log"
python -u -m paddle.distributed.launch --gpus ${gpu_ids} --log_dir ${log_dir} \
	run_simcse.py \
	--mode "train" \
	--model_name_or_path "rocketqa-zh-dureader-query-encoder" \
	--max_seq_length "128" \
	--output_emb_size "32" \
	--train_set_file "your train_set path" \
	--eval_set_file "your dev_set path" \
	--save_dir "checkpoints" \
	--log_dir ${log_dir} \
	--save_steps "5000" \
	--eval_steps "100" \
	--batch_size "32" \
	--epochs "3" \
	--learning_rate "3e-5" \
	--weight_decay "0.01" \
	--warmup_proportion "0.01" \
	--dropout "0.1" \
	--margin "0.0" \
	--scale "20" \
	--dup_rate "0.0" \
	--seed "0" \
	--device "gpu"

gpu_ids=0
export CUDA_VISIBLE_DEVICES=${gpu_ids}

log_dir="log_test"
python -u -m paddle.distributed.launch --gpus ${gpu_ids} --log_dir ${log_dir} \
	run_simcse.py \
	--mode "eval" \
	--model_name_or_path "rocketqa-zh-dureader-query-encoder" \
	--max_seq_length "128" \
	--output_emb_size "32" \
	--eval_set_file "../../../dataset/100w/test_v1.txt" \
	--ckpt_dir "./checkpoints_simcse_rocket_webdata_wr0" \
	--batch_size "8" \
	--dropout "0.1" \
	--margin "0.0" \
	--scale "20" \
	--dup_rate "0.3" \
	--seed "0" \
	--device "gpu"

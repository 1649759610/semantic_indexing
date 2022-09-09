export CUDA_VISIBLE_DEVICES=0

python 	run_simcse.py \
	--mode "infer" \
    --model_name_or_path "rocketqa-zh-dureader-query-encoder" \
	--max_seq_length "128" \
	--output_emb_size "32" \
	--infer_set_file ./data/hard_case.txt \
	--ckpt_dir ./checkpoints/best \
	--batch_size "16" \
	--dropout "0.1" \
	--margin "0.0" \
	--scale "20" \
	--dup_rate "0.3" \
	--seed "0" \
	--device "gpu"
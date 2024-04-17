CUDA_VISIBLE_DEVICES=2,3 python semantic_align.py \
--source_file='' \
--target_file='' \
--model_dir='emotion2vec-main/upstream' \
--checkpoint_dir='emotion2vec-main/pretrain/google_cloud/emotion2vec_base.pt' \
--granularity='frame' \
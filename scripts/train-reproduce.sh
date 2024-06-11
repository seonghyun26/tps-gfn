CUDA_VISIBLE_DEVICES=$1 python src/train.py \
  --date reproduce \
  --config configs/alanine/reproduce.yaml
  # --config configs/alanine/reproduce.yaml
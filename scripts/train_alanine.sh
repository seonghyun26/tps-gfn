CUDA_VISIBLE_DEVICES=$1 python src/train.py \
  --config $2
  # --config configs/alanine/reproduce.yaml
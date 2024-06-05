current_date=$(date "+%m%d-%H%M%S")
# echo $current_date

CUDA_VISIBLE_DEVICES=$1 python src/train.py \
  --date $current_date \
  --config $2
  # --config configs/alanine/reproduce.yaml
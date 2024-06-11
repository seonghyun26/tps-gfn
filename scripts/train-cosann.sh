current_date=$(date "+%m%d-%H%M%S")

echo ">> Training cosann"

for seed in {0..7}; do
  echo ">> Training seed $seed"
  CUDA_VISIBLE_DEVICES=$seed python src/train.py \
    --date $current_date \
    --config configs/alanine/tune-cosann.yaml \
    --seed $seed \
    --bias_scale 20 &
  sleep 1
  done
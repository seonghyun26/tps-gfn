current_date=$(date "+%m%d-%H%M%S")
# echo $current_date
echo ">> Training alanine"
sleep 0.5

for seed in {0..7}; do
  echo "Training seed $seed"
  CUDA_VISIBLE_DEVICES=$seed python src/train.py \
    --date $current_date \
    --seed $seed \
    --config configs/alanine/tune-$1.yaml &
    # --config configs/alanine/reproduce.yaml
  sleep 1
done
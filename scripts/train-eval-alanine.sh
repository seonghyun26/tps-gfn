current_date=$(date "+%m%d-%H%M%S")

echo ">> Training alanine, $current_date"
sleep 1

for seed in 0 1; do
  echo "Training seed $seed"
  CUDA_VISIBLE_DEVICES=$seed python src/train.py \
    --config configs/alanine/tune-$1.yaml \
    --date $current_date \
    --seed $seed 
  sleep 1

  echo "Evaluating seed $seed"
  CUDA_VISIBLE_DEVICES=$seed python src/eval.py \
    --config configs/alanine/tune-$1.yaml \
    --date $current_date \
    --seed $seed
done
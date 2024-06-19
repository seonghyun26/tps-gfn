current_date=$(date "+%m%d-%H%M%S")
echo ">> Training alanine"
sleep 0.5

for seed in 1; do
  echo "Training seed $seed"
  CUDA_VISIBLE_DEVICES=2 python src/train.py \
    --date $current_date \
    --project multi-goal \
    --config configs/alanine/debug.yaml \
    --seed $seed \
    --wandb
  sleep 1
done
seed=0
date=unit
project=debug
config=configs/alanine/debug.yaml

echo "Unit test on training"
CUDA_VISIBLE_DEVICES=4 python src/train.py \
    --date $current_date \
    --project $porject \
    --config $config  \
    --seed $seed

echo "Unit test on Evaluation"
CUDA_VISIBLE_DEVICES=1 python src/eval.py \
  --type eval \
  --date $current_date \
  --project $porject \
  --config $config  \
  --seed $seed

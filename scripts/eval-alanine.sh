echo ">> Evaluating alanine"
sleep 0.5

CUDA_VISIBLE_DEVICES=1 python src/eval.py \
  --type eval \
  --project multi-goal \
  --config configs/alanine/debug.yaml \
  --date 0619-034844 \
  --seed 1

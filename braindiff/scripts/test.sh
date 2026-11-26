#!/bin/bash
# Test-set generation with the reported decode (greedy + output-only repetition penalty 1.1) and Table 2 scoring.
#   scripts/test.sh [checkpoint] [gpus] [out dir]     default: checkpoints/braindiff.pt 0,1,2,3 out/ours
# Env as scripts/train.sh (BRAINDIFF_DATA, BRAINDIFF_SCORE_PYTHON, RADDELTA_CKPT).
set -euo pipefail
CK=${1:-checkpoints/braindiff.pt}; GPUS=${2:-0,1,2,3}; OUT=${3:-out/ours}
NP=$(echo "$GPUS" | tr ',' '\n' | grep -c .); SPY=${BRAINDIFF_SCORE_PYTHON:-python}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} TOKENIZERS_PARALLELISM=false
mkdir -p $OUT
for sp in test external_test; do
  CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node=$NP -m braindiff.eval.generate --ckpt $CK --split $sp \
      --repetition_penalty 1.1 --out $OUT/gen_${sp}.jsonl > $OUT/gen_${sp}.log 2>&1
  echo "=== generated $sp: $(wc -l < $OUT/gen_${sp}.jsonl) pairs"
done
i=0; for sp in test external_test; do
  CUDA_VISIBLE_DEVICES=$(echo $GPUS | cut -d, -f$((i + 1))) $SPY -m braindiff.eval.score $OUT/gen_${sp}.jsonl \
      > $OUT/score_${sp}.log 2>&1 & i=$((i + 1)); done; wait
echo "=== scored: $(ls $OUT/score_*.npz $OUT/*.agree.npz $OUT/perpair_*.parquet | wc -l) / 6 files"

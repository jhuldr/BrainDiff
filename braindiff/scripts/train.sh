#!/bin/bash
# Stage 1-D, its epoch selection, Stage 2 (status-first), then Stage 2 epoch selection.
#   scripts/train.sh [gpus]          e.g. scripts/train.sh 0,1,2,3     (4 GPUs: global batch 32 = 4 x 8)
# Env: BRAINDIFF_DATA (tables, default ./data); BRAINDIFF_SCORE_PYTHON (python of the scoring env, radgraph +
# transformers < 5, with braindiff installed); RADDELTA_CKPT (Rad-Δ student). The in-loop eval scores in the background.
set -euo pipefail
GPUS=${1:-0,1,2,3}; NP=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPUS
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} TOKENIZERS_PARALLELISM=false
SPY=${BRAINDIFF_SCORE_PYTHON:-python}
CFG=$(python -c "import braindiff, os; print(os.path.join(os.path.dirname(braindiff.__file__), 'configs'))")
say() { echo "=== [$(date '+%a %H:%M')] $*"; }
mkdir -p logs

say "Stage 1-D"
torchrun --standalone --nproc_per_node=$NP -m braindiff.training.train_stage1 --config $CFG/stage1.yaml > logs/stage1.log 2>&1
K=$(python -m braindiff.training.train_stage1 --select checkpoints/stage1)
say "Stage 1-D selected epoch $K (max val auc_nll)"
sed "s#^init_vision: .*#init_vision: ./checkpoints/stage1/epoch$K.pt#" $CFG/stage2.yaml > logs/stage2.yaml

say "Stage 2"
torchrun --standalone --nproc_per_node=$NP -m braindiff.training.train_stage2 --config logs/stage2.yaml > logs/stage2.log 2>&1
D=checkpoints/stage2; LAST=$(python -c "import yaml; print(yaml.safe_load(open('logs/stage2.yaml'))['epochs'])")
until [ -s $D/score_epoch$LAST.npz ]; do sleep 60; done          # in-loop scoring runs in the background

say "Stage 2 selection: top 3 by subset macro-MCC -> full val"
for E in $(python -m braindiff.eval.generate pick $D); do
  torchrun --standalone --nproc_per_node=$NP -m braindiff.eval.generate --ckpt $D/epoch$E.pt --init ./checkpoints/stage1/epoch$K.pt \
      --split val --out $D/gen_val_epoch$E.jsonl > logs/select_epoch$E.log 2>&1
  CUDA_VISIBLE_DEVICES=${GPUS%%,*} $SPY -m braindiff.eval.score $D/gen_val_epoch$E.jsonl > logs/score_val_epoch$E.log 2>&1
done
E=$(python -m braindiff.eval.generate pick $D --full)
say "selected Stage 2 epoch $E"
python scripts/strip_release_checkpoint.py --stage2 $D/epoch$E.pt --stage1 checkpoints/stage1/epoch$K.pt --out checkpoints/braindiff.pt

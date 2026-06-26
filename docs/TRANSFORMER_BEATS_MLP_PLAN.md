# Transformer Beats MLP Plan

## Highest-impact ideas (in order)

### 1) Tune Attention separately (don’t share MLP hyperparams)

Transformers usually want:
- lower LR
- more warmup
- slightly stronger gradient clipping
- tuned entropy coefficient

If both models use identical PPO settings, MLP often wins by stability, not capacity.

### 2) Select best checkpoint, not just final checkpoint

Your run can end after a late-training dip.
Save/evaluate periodically and choose the checkpoint with best eval mean (or best CVaR if risk-aware).
This alone often flips outcomes.

### 3) Give Attention longer or staged training

Attention can lag early and catch up later.
Try:
- Stage A quick stabilization (short horizon)
- Stage B full-budget finetune with lower LR

Same total budget, better optimization path.

### 4) Make the task actually reward context modeling

Your current attention variant is single-snapshot; if signal is mostly local, MLP has an edge.
Add short history input (or richer temporal features) so attention’s inductive bias matters.

### 5) Run multi-seed model selection

If your target is “Transformer should beat MLP,” optimize against average over 3–5 seeds, not one seed.
This prevents chasing lucky/unlucky seed artifacts.

---

## Concrete recipe to implement first (minimal disruption)

Keep current full pipeline.
For Transformer only:
- LR: reduce by 2–4x
- add warmup (e.g., first 5–10% updates)
- gradient clip a bit tighter
- slightly higher entropy early, anneal later

Evaluate every fixed interval; keep best checkpoint.
Then compare best-attention vs best-MLP under same eval protocol.

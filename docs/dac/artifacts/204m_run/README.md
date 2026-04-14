# 204M LeanFormer Training Run — Artifacts

This directory holds the training-governance artifacts from the 204M-parameter LeanFormer run reported in Section 5.8 of the DAC paper. The run trained a 204M-parameter LeanFormer (805M dense equivalent) on a reasoning corpus for 7,228 optimizer steps (one epoch) on an NVIDIA L4 GPU, wall clock 140.9 hours.

Large binary checkpoints (`pytorch_model.bin`, `training_state.pt`, step-*/ snapshots, optimizer state — several GB total) are **not** included here. The governance-machinery claims in the paper are verifiable from the files below without them. Readers who need the checkpoints for independent replication of the 39M or 204M evaluations can request them from the author.

## Files

| File | Size | What It Is |
|------|------|------------|
| `audit.jsonl` | 828K | SHA-256 hash-chained audit log. 722 records, one per 10 training steps from step 10 to step 7,220. Each record contains gradient norms per parameter group, convergence states, budget allocations, active hierarchy levels, loss, timestamp, and `record_hash` / `prev_hash` for chain verification. This is the primary artifact that demonstrates zero governance violations across the run. |
| `training_log.json` | 40K | Step-by-step training metrics: loss, learning rate, gradient clipping, per-group gradient EMA, validation triggers. Use this for plotting training dynamics. |
| `training_meta.json` | 1K | Final metadata: `optimizer_step: 7228`, `val_ppl: 57.58` (the best-checkpoint val PPL @ step 2,000 re-evaluated), `best_val_loss`. |
| `validation_results.json` | 12K | Per-checkpoint validation metrics recorded during the run (val loss and val PPL at step 500/1000/1500/2000/... across the full run). |
| `best_checkpoint_validation.json` | 12K | Standalone re-validation of the step-2,000 checkpoint (the best-PPL checkpoint) against a held-out set, including generation samples referenced in Section 5.8. |
| `leanformer_config.json` | 1K | The exact `LeanFormerConfig` used to instantiate the model (d_model=1536, n_heads=24, n_layers=20, attention_rank=192, ff_rank=192, screening_rank=48, top_k=96, ff_sparsity_target=0.8, dropout=0.1, delta_rank=4). |
| `model_hash.txt` | <1K | SHA-256 of the trained base model weights. This is the hash that every `.delta` file produced against this model must pass at load time (DFS v2.0 contract). |
| `configs/reasoning_core_204m.yaml` | — | Training config (optimizer, scheduler, batch size, seq len, gradient accumulation, convergence thresholds, budget policy). |
| `configs/parameter_groups.json` | — | The parameter-group registry used for this run. Every tensor in the 204M model is assigned to one of eight groups (embeddings, attention_routing, gates, layer_norms, attention_output, ff_projections, output_head, exit_classifier), each tagged with a hierarchy level L0-L3. The fnmatch patterns are config-independent and work for any model size. |
| `facts.json` | 20K | The fact bank used by the Knowledge Plane (not the training corpus — the training corpus is several GB and is not published here). |
| `requirements_frozen.txt` | 2K | Exact pip-freeze of the training environment. |
| `training.log` | 20K | Text training log (stdout). Human-readable timeline. |

## Verifying the Audit Chain

The audit log is tamper-evident via SHA-256 hash chaining. Each record has a `record_hash` computed over its content plus the previous record's `record_hash`. The first record's `prev_hash` is all zeros. Any modification or reordering breaks the chain.

```python
import hashlib
import json

prev = "0" * 64
with open("audit.jsonl") as f:
    for i, line in enumerate(f):
        rec = json.loads(line)
        assert rec["prev_hash"] == prev, f"chain broken at record {i}"
        # record_hash is computed by leanformer.training.audit.AuditSink;
        # see leanformer/training/audit.py for the canonical implementation.
        prev = rec["record_hash"]
print("chain intact across", i + 1, "records")
```

The reference implementation lives in `leanformer/training/audit.py` in the main repo.

## Key Numbers (All Derivable From These Files)

- 722 audit records across 7,228 training steps
- Zero budget invariant violations (`sum(allocations) <= 1.0` at every record)
- 18 governor state transitions across 8 parameter groups, all valid (no skipped states)
- L0 active from step 0; L1/L2 activated at steps 200/400 (B=0 initialization artifact — see Section 5.8.1 of the paper); L3 activated at step 2,773 via genuine post-learning convergence
- Best validation perplexity: 57.6 @ step 2,000
- Final validation perplexity: 1,463.9 (overfit; all invariants held throughout)
- Final convergence states: embeddings COOLING, attention_routing CONVERGED, gates CONVERGED, layer_norms CONVERGED, attention_output COOLING, ff_projections COOLING, output_head COOLING, exit_classifier CONVERGED
- One `COOLING → ACTIVE` regression in `attention_output` demonstrating governor hysteresis

## Relationship to the Paper

Figures 1-3 in `docs/dac/figures/` are generated from these artifacts:

- **Figure 1 (training dynamics)** — `training_log.json` + `validation_results.json`
- **Figure 2 (gradient norm heatmap)** — `audit.jsonl` (per-group gradient norms per record)
- **Figure 3 (governor state timeline)** — `audit.jsonl` (per-group convergence states per record)

The B=0 observational degeneracy (Section 5.8.1) is directly visible in the first 40 records of `audit.jsonl`: the `gradient_norms` field shows near-zero values for most groups through step 400, at which point the `output_head` group's gradient jumps from 0.0 to 0.88 in a single step when L2 activates.

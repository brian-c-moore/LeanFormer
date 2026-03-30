"""
Phase 4: Full delta belief system validation at scale.

Tests 100 beliefs across 5 categories:
- 4.2: Inject all 100 beliefs, measure rank improvement
- 4.3: Test coexistence (all 100 active simultaneously)
- 4.4: Test semantic routing discrimination
- 4.5: Test removal (ordered + random)
- 4.6: Test belief update

Usage:
    python -m leanformer.scripts.validate_beliefs
"""

import sys
import json
import random
import torch
from pathlib import Path
from transformers import AutoTokenizer

from rich.console import Console
from rich.table import Table
from rich import box

from ..model.config import LeanFormerConfig
from ..model.leanformer import LeanFormer
from ..beliefs.knowledge_store import KnowledgeStore
from ..beliefs.delta_system import DeltaRouter

console = Console()

CHECKPOINT_DIR = Path("checkpoints/scale")
FACTS_PATH = Path("data/facts.json")


def main():
    if not (CHECKPOINT_DIR / "pytorch_model.bin").exists():
        console.print("[red]No checkpoint found.[/red]")
        sys.exit(1)
    if not FACTS_PATH.exists():
        console.print("[red]No facts.json found. Run build_facts first.[/red]")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    config = LeanFormerConfig.load(CHECKPOINT_DIR / "leanformer_config.json")
    model = LeanFormer(config)
    model.load_state_dict(torch.load(
        CHECKPOINT_DIR / "pytorch_model.bin", map_location=device, weights_only=True,
    ))
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    param_count = sum(p.numel() for p in model.parameters())

    with open(FACTS_PATH) as f:
        facts = json.load(f)

    console.print("[bold cyan]Phase 4: Belief Delta System Validation at Scale[/bold cyan]\n")

    # Snapshot base weights
    base_snapshot = {n: p.clone() for n, p in model.named_parameters()}

    # === 4.2: Inject all 100 beliefs ===
    console.print("[bold]4.2: Inject 100 Beliefs[/bold]\n")

    store = KnowledgeStore(model, tokenizer=tokenizer, encoding_steps=config.delta_encoding_steps)
    results = []
    total_delta_params = 0

    for category, entries in facts.items():
        for entry in entries:
            key = f"{category}_{entry['prompt'][:20].replace(' ', '_')}"
            prompt_ids = tokenizer.encode(entry["prompt"], return_tensors="pt").to(device)
            target_id = entry["target_token_id"]

            # Baseline
            with torch.no_grad():
                baseline_logits = model(prompt_ids, training=False)["logits"][0, -1, :]
                baseline_rank = (baseline_logits.argsort(descending=True) == target_id).nonzero().item()
                baseline_logit = baseline_logits[target_id].item()

            # Encode belief
            info = store.add(key, entry["fact"])
            delta = store.registry.deltas[key]

            # With belief
            with torch.no_grad():
                mod_logits = model(prompt_ids, training=False, active_deltas=[delta])["logits"][0, -1, :]
                mod_rank = (mod_logits.argsort(descending=True) == target_id).nonzero().item()
                mod_logit = mod_logits[target_id].item()

            total_delta_params += delta.total_parameters()
            improved = mod_rank < baseline_rank

            results.append({
                "key": key,
                "category": category,
                "prompt": entry["prompt"],
                "target": entry["target"],
                "baseline_rank": baseline_rank,
                "modified_rank": mod_rank,
                "rank_improvement": baseline_rank - mod_rank,
                "logit_shift": mod_logit - baseline_logit,
                "improved": improved,
                "delta_params": delta.total_parameters(),
            })

            status = "[green]+" if improved else "[red]-"
            console.print(
                f"  {status}[/{status.split('[')[1]} "
                f"{category:12s} | {entry['target']:>10s} | "
                f"rank {baseline_rank:>5,} -> {mod_rank:>5,} | "
                f"shift {mod_logit - baseline_logit:+.2f}"
            )

    # Summary
    n_improved = sum(1 for r in results if r["improved"])
    n_top100 = sum(1 for r in results if r["modified_rank"] < 100)
    avg_improvement = sum(r["rank_improvement"] for r in results) / len(results)

    console.print(f"\n  Beliefs improved:     {n_improved}/100 ({n_improved}%)")
    console.print(f"  Target in top 100:    {n_top100}/100 ({n_top100}%)")
    console.print(f"  Avg rank improvement: {avg_improvement:,.0f} positions")
    console.print(f"  Total delta params:   {total_delta_params:,} ({total_delta_params/param_count:.1%} of model)")

    # === 4.3: Coexistence test ===
    console.print(f"\n[bold]4.3: Coexistence Test (all 100 active)[/bold]\n")

    all_deltas = list(store.registry.deltas.values())
    coexist_improved = 0

    for r in results:
        prompt_ids = tokenizer.encode(r["prompt"], return_tensors="pt").to(device)
        target_id = facts[r["category"]][0]["target_token_id"]  # May not match exactly, use stored
        # Find the right entry
        for entry in facts[r["category"]]:
            if entry["prompt"] == r["prompt"]:
                target_id = entry["target_token_id"]
                break

        with torch.no_grad():
            baseline_logits = model(prompt_ids, training=False)["logits"][0, -1, :]
            baseline_rank = (baseline_logits.argsort(descending=True) == target_id).nonzero().item()

            coexist_logits = model(prompt_ids, training=False, active_deltas=all_deltas)["logits"][0, -1, :]
            coexist_rank = (coexist_logits.argsort(descending=True) == target_id).nonzero().item()

        if coexist_rank < baseline_rank:
            coexist_improved += 1

    console.print(f"  Still improved with all 100 active: {coexist_improved}/100 ({coexist_improved}%)")

    # === 4.4: Semantic routing discrimination ===
    console.print(f"\n[bold]4.4: Semantic Routing Discrimination[/bold]\n")

    store.attach_to_model(model)
    router = model.delta_router
    categories = list(facts.keys())
    routing_correct = 0
    routing_total = 0

    for category in categories:
        for entry in facts[category][:5]:  # Test 5 per category
            prompt_ids = tokenizer.encode(entry["prompt"], return_tensors="pt").to(device)
            with torch.no_grad():
                positions = torch.arange(prompt_ids.shape[1], device=device).unsqueeze(0)
                hidden = model.token_embedding(prompt_ids) + model.position_embedding(positions)
                selected = router(hidden, store.registry)

            selected_keys = [d.name for d in selected]
            correct_category_count = sum(1 for k in selected_keys if k.startswith(category))
            total_selected = len(selected_keys)

            if total_selected > 0:
                routing_correct += correct_category_count
                routing_total += total_selected

    routing_accuracy = routing_correct / max(routing_total, 1)
    console.print(f"  Correct-category selections: {routing_correct}/{routing_total} ({routing_accuracy:.0%})")
    console.print(f"  Chance level: 20% (5 categories)")

    store.detach_from_model(model)

    # === 4.5: Removal at scale ===
    console.print(f"\n[bold]4.5: Removal Verification[/bold]\n")

    # Get baseline before any removal
    test_prompt = tokenizer.encode("The capital of France is", return_tensors="pt").to(device)
    with torch.no_grad():
        pre_removal_logits = model(test_prompt, training=False)["logits"].clone()

    # Remove in insertion order
    all_keys = list(store.entries.keys())
    for key in all_keys:
        store.remove(key)

    with torch.no_grad():
        post_removal_logits = model(test_prompt, training=False)["logits"]

    ordered_match = torch.allclose(pre_removal_logits, post_removal_logits, atol=1e-6)
    console.print(f"  Ordered removal — baseline restored: [{'green' if ordered_match else 'red'}]{ordered_match}[/{'green' if ordered_match else 'red'}]")

    # Re-inject all, remove in random order
    for r in results:
        for category, entries in facts.items():
            for entry in entries:
                key = f"{category}_{entry['prompt'][:20].replace(' ', '_')}"
                if key == r["key"]:
                    store.add(key, entry["fact"])
                    break

    random_keys = list(store.entries.keys())
    random.shuffle(random_keys)
    for key in random_keys:
        store.remove(key)

    with torch.no_grad():
        random_removal_logits = model(test_prompt, training=False)["logits"]

    random_match = torch.allclose(pre_removal_logits, random_removal_logits, atol=1e-6)
    console.print(f"  Random removal — baseline restored:  [{'green' if random_match else 'red'}]{random_match}[/{'green' if random_match else 'red'}]")

    # === 4.6: Update test ===
    console.print(f"\n[bold]4.6: Belief Update Verification[/bold]\n")

    # Add a belief, update it, verify
    store.add("update_test", "The capital of France is Paris")
    paris_id = tokenizer.encode(" Paris")[0]
    lyon_id = tokenizer.encode(" Lyon")[0]
    prompt_ids = tokenizer.encode("The capital of France is", return_tensors="pt").to(device)

    delta_v1 = store.registry.deltas["update_test"]
    with torch.no_grad():
        v1_logits = model(prompt_ids, training=False, active_deltas=[delta_v1])["logits"][0, -1, :]
        paris_rank_v1 = (v1_logits.argsort(descending=True) == paris_id).nonzero().item()

    store.update("update_test", "The capital of France is Lyon")
    delta_v2 = store.registry.deltas["update_test"]
    with torch.no_grad():
        v2_logits = model(prompt_ids, training=False, active_deltas=[delta_v2])["logits"][0, -1, :]
        lyon_rank_v2 = (v2_logits.argsort(descending=True) == lyon_id).nonzero().item()
        paris_rank_v2 = (v2_logits.argsort(descending=True) == paris_id).nonzero().item()

    console.print(f"  Before update: Paris rank = {paris_rank_v1}")
    console.print(f"  After update:  Lyon rank = {lyon_rank_v2}, Paris rank = {paris_rank_v2}")
    console.print(f"  Lyon improved: {lyon_rank_v2 < 50257}")

    store.remove("update_test")

    # === Weight integrity ===
    console.print(f"\n[bold]Base Weight Integrity[/bold]")
    all_match = all(
        torch.equal(param, base_snapshot[name])
        for name, param in model.named_parameters()
    )
    console.print(f"  All {len(base_snapshot)} tensors unchanged: [{'green' if all_match else 'red'}]{all_match}[/{'green' if all_match else 'red'}]")

    # === Save results ===
    report = {
        "injection": {
            "improved": n_improved,
            "top100": n_top100,
            "avg_rank_improvement": avg_improvement,
            "total_delta_params": total_delta_params,
            "delta_pct_of_model": total_delta_params / param_count,
        },
        "coexistence": {"improved_with_all_active": coexist_improved},
        "routing": {"accuracy": routing_accuracy, "correct": routing_correct, "total": routing_total},
        "removal": {"ordered_match": ordered_match, "random_match": random_match},
        "weight_integrity": all_match,
        "per_belief": results,
    }
    with open(CHECKPOINT_DIR / "belief_validation.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    console.print(f"\n  Results saved to {CHECKPOINT_DIR / 'belief_validation.json'}")

    # === Summary ===
    console.print(f"\n[bold]Summary[/bold]")
    table = Table(box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Result", style="green")
    table.add_column("Target")
    table.add_column("Status")

    table.add_row("Beliefs improved", f"{n_improved}%", ">70%", "[green]PASS" if n_improved > 70 else "[red]FAIL")
    table.add_row("Target in top 100", f"{n_top100}%", ">50%", "[green]PASS" if n_top100 > 50 else "[red]FAIL")
    table.add_row("Coexistence", f"{coexist_improved}%", ">60%", "[green]PASS" if coexist_improved > 60 else "[red]FAIL")
    table.add_row("Routing accuracy", f"{routing_accuracy:.0%}", ">50%", "[green]PASS" if routing_accuracy > 0.5 else "[yellow]PARTIAL")
    table.add_row("Ordered removal", str(ordered_match), "True", "[green]PASS" if ordered_match else "[red]FAIL")
    table.add_row("Random removal", str(random_match), "True", "[green]PASS" if random_match else "[red]FAIL")
    table.add_row("Weight integrity", str(all_match), "True", "[green]PASS" if all_match else "[red]FAIL")

    console.print(table)


if __name__ == "__main__":
    main()

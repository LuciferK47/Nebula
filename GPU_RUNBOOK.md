# GPU Runbook — what to run on the RTX 4050

Ordered cheapest-and-most-likely-to-fail first, so you find breakage in seconds
rather than 15 minutes into a sweep.

Everything below runs from the repo root with the venv active:

```powershell
cd Nebula
.venv\Scripts\Activate.ps1     # Windows PowerShell
# source .venv/bin/activate    # Linux / macOS
```

---

## Step 0 — Confirm the environment (10 seconds)

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True` and `NVIDIA GeForce RTX 4050 Laptop GPU`. If `False`, nothing below
is meaningful — fix CUDA first.

---

## Step 1 — Full test suite (~30 seconds)

```powershell
pytest tests -v
```

**Expect 93 passed.** That is the original 75 plus 18 new ones:

| File | Tests | Covers |
| :--- | ---: | :--- |
| `test_dram_pool_charging.py` | 6 | the DRAM→HBM double-charge fix |
| `test_emulation_mode_parity.py` | 3 | CXL emulation mode must not affect placement |
| `test_metrics_snapshot.py` | 3 | windowed hit-rate maths for S5 |
| `test_degenerate_budgets.py` | 6 | degenerate tier budgets (CPU-only half of S10) |

`test_real_module_transfer.py` (4 tests) needs torch and was never runnable on the
Mac, so it is the one file whose result is genuinely new information here.

If anything fails, stop and send me the output — don't proceed to the sweeps.

---

## Step 2 — Edge cases FIRST (~3 minutes) ⭐

```powershell
python scripts/run_scenarios.py --scenario s10
```

Run this **before** any measurement scenario. It deliberately tries to break the
system, so it's the cheapest way to find out whether the full sweep will survive.

Read `results/scenarios/s10.json` and check three things:

1. **`summary.failed` should be 0.** `expected_fail` of 2 is correct — `e4` and
   `e6` exist to confirm we refuse impossible budgets cleanly.
2. **`model_contaminated` must be `false`.** If it's `true`, a deliberate failure
   left the shared base model half-migrated between devices, and every scenario
   after it in the same process is untrustworthy. Re-run the others individually.
3. **`e10_patch_unpatch_leak.result.leak_suspected` must be `false`.** The full
   sweep builds 40+ wrappers in one process; a per-construction leak would
   accumulate and OOM mid-run.

Any `"status": "FAIL"` — send me the `error` string for that case.

---

## Step 3 — The CXL sensitivity sweep (~2 minutes) ⭐

```powershell
python scripts/run_scenarios.py --scenario s3
```

This is the scenario that decides whether you need DRAMSim3 at all. In
`results/scenarios/s3.json`:

- **`placement_stable_across_bandwidth_sweep` must be `true`.** It means a 16×
  swing in assumed CXL bandwidth left hit rate untouched — i.e. the emulation is
  a pure timing knob and cannot be silently driving your conclusions.
- Compare `tokens_per_second` across the five bandwidth runs and the
  `disabled` / `latency_only` runs. **If the ranking of configurations is stable
  across all of them, your architectural claim does not depend on CXL calibration
  accuracy** — that is the result to put in the paper, and it is a stronger claim
  than a DRAMSim3 cross-check would give you.
- If the ranking *flips* anywhere in that sweep, tell me. It means the claim is
  bandwidth-assumption-dependent and needs narrowing before publication.

It also drops `s3_trace.dramsim3`, `s3_trace_gem5.txt` and `s3_trace.csv` in
`results/scenarios/` — ship those as an artifact so reviewers can cross-check
offline without you taking on a simulator dependency.

---

## Step 4 — The full suite (~15 minutes)

```powershell
python scripts/run_scenarios.py --scenario all
```

Runs S1–S7, S9, then S10 last. Prints per-scenario elapsed time and a total at
the end. Writes one JSON per scenario into `results/scenarios/`.

| Scenario | ~Time | What it gives you |
| :--- | ---: | :--- |
| `s1` capacity cliff | 2.5 min | **the headline chart** — throughput/hit-rate vs. HBM budget |
| `s2` execution modes | 2.5 min | the properly-ablated version of the 1.78× claim |
| `s3` CXL sensitivity | 2 min | the "is your CXL real?" defense |
| `s4` domain shift | 1 min | how far OOD placement degrades (honest weakness probe) |
| `s5` long horizon | 1.5 min | **best live visual** — 500 tokens, rolling-window time series |
| `s6` correctness | 2 min | prefix-match / divergence / logit delta vs. full VRAM |
| `s7` batch scaling | 2 min | where hybrid's edge erodes as batching kills sparsity |
| `s9` overlap proof | 1 min | CUDA-event proof that transfers overlap compute |
| `s10` edge cases | 3 min | robustness (runs last, by design) |

Any single scenario can be re-run on its own with `--scenario s5`, which is what
you want during a live demo.

### Two things to sanity-check in the output

- **`s5`** — watch whether windowed `hit_rate` converges or drifts downward over
  the 500 tokens. Drifting means the LFU decay is thrashing at that budget.
  Note this one re-runs `generate()` per 50-token window, so the growing prompt
  gets reprocessed each time; if it overruns 1.5 min noticeably, that's why, and
  it's fixable by widening the window.
- **`s6`** — `exact_prefix_match_length` will read *lower* than the old
  stress-test's token-match percentage. That is correct, not a regression: the
  old check compared token IDs elementwise and averaged away a single early
  divergence, this one reports where output actually first departs from the
  full-VRAM reference.

---

## Step 5 — Scale validation on the A100 / L40S (~10 minutes)

Run on the bigger GPU, not the 4050. Point it at a larger MoE and constrain the
HBM budget to roughly the same *fraction* of expert bytes as the 4050 runs (~36%),
so the comparison is like-for-like:

```powershell
python scripts/run_scenarios.py --scenario s1 --model-id Qwen/Qwen1.5-MoE-A2.7B --tokens 25
python scripts/run_scenarios.py --scenario s2 --model-id Qwen/Qwen1.5-MoE-A2.7B --tokens 25
```

This is the single highest-value addition for the paper — it shows the mechanism
isn't an artifact of a 4×0.5B toy model. The 4050-only story is materially weaker
without it.

Note the budgets inside the scenarios are currently hardcoded for the small model
(600/900MB etc.). For the big model you'll want them scaled up; tell me the
expert-set size that `_patch_moe_layers` reports and I'll parameterize them.

---

## Step 6 — Serve it to the frontend

```powershell
python scripts/serve.py
```

`/api/historical_results` now returns a `scenarios` key containing every JSON in
`results/scenarios/`, keyed by scenario id. No new endpoint, no frontend change
required to fetch it.

---

## Two things to keep honest on the frontend

1. **Never render `benchmark_results.json`'s `tokens_per_second` next to a
   scenario's.** The former is a closed-form analytical model with a hardcoded
   85ms/token constant (`benchmark_runner.py`); the latter is measured wall
   clock. Every scenario result carries `computation_type` and
   `transfer_latency_type` badges — use them.
2. **Label the budget "expert cache budget", not VRAM.** At a 300MB budget peak
   VRAM is still ~3.8GB — the base model, KV cache and allocator dominate.
   Calling it "runs in 300MB" reads as a false claim.

---

## If you hit a wall

Send me the failing command plus the JSON (or traceback). The two bugs already
fixed — a crash in the CXL→DRAM→HBM staging hop, and silent weight loss when all
tiers were oversubscribed — were both found by writing S10, so S10 failing on
real hardware is informative, not embarrassing.

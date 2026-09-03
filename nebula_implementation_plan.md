# MemTier-MoE — Implementation Plan v3

## Fixes: latency injection, scope creep, build order, tooling assumptions, test coverage.

---

## Critical Fixes from Review

### 🔴 `time.sleep(350ns)` is broken

Python `time.sleep` has ~1ms granularity + GIL. 350ns delay becomes ~1-15ms (10,000-40,000× too large).

**Fix**: Busy-wait spin loop against `time.perf_counter_ns()`:

```python
def inject_latency_ns(target_ns: int):
    """Spin-wait for sub-microsecond latency injection."""
    start = time.perf_counter_ns()
    while (time.perf_counter_ns() - start) < target_ns:
        pass
```

> [!IMPORTANT]
> **Accepted approximation**: Python's `while` loop + `perf_counter_ns()` call overhead is plausibly in the same order of magnitude as 350ns itself. This means injected latency will be "roughly right" (probably 1–3× the target), not nanosecond-precise. This is fine for a functional PoC — what matters is that the relative ordering **HBM < DRAM < CXL** is preserved and measurable, not that we hit exactly 350ns. We state this explicitly rather than claiming exact calibration.

**Optional/stretch**: GPU-side latency injection via `numba.cuda.jit` kernel (NOT `torch.cuda.jit`, which doesn't exist). Requires `numba` dependency + dealing with its compilation caching and kernel launch overhead — not worth the risk during the hackathon unless the CPU spin-wait proves insufficient. Leaving this as a documented future improvement.

### 🟡 CUDA stream overlap must be verified, not assumed

Add explicit profiler verification step in M2. If overlap isn't happening, nothing else matters.

**Primary tool**: `torch.profiler` (pure Python, works everywhere including WSL2):
```python
with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    run_overlap_test()
print(prof.key_averages().table())
prof.export_chrome_trace("overlap_trace.json")  # inspect in chrome://tracing
```

**Secondary tool**: `nsys` (richer GPU timeline, but **must be verified in M0** — see below).

### 🟡 Environment setup is a time bomb

Model download (~7GB), `auto-gptq` build, CUDA toolkit matching — do this FIRST, before any code.

---

## Milestone Structure (replaces 7 sequential phases)

### Milestone 0: Environment Bootstrap [Do immediately]

```bash
# WSL2 Ubuntu 24
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers safetensors datasets numpy pandas matplotlib pyyaml pytest tqdm

# Quantization — pick ONE, test it builds
pip install auto-gptq  # OR: pip install autoawq  # OR: pip install bitsandbytes

# Download model ONCE, cache it
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-MoE-A2.7B-GPTQ-Int4', device_map='cpu')"

# Verify CUDA works in WSL2
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Verify profiling tools — nsys is flaky under WSL2 (driver/permission issues vary by build)
# Test it NOW so M2 doesn't depend on a tool that might not work
nsys profile -o /tmp/test_nsys python -c "import torch; x=torch.randn(100,device='cuda'); y=x@x"
# If nsys fails: torch.profiler is the fallback (already verified by torch import above)
# If nsys works: great, use it for M2 overlap verification
```

**Exit criteria**: Model loaded, CUDA confirmed, no import errors, profiling tool confirmed (nsys OR torch.profiler).

---

### Milestone 1: Thin Vertical Slice [MUST-HAVE for demo]

**Goal**: 2-tier system (HBM + DRAM), simple LFU cache, working inference loop, no prefetcher. Demoable end-to-end.

```
Files (12):
  memtier_moe/
  ├── core/types.py, config.py, metrics.py
  ├── introspect/weight_profiler.py, routing_tracer.py
  ├── memory/expert_metadata.py, tier_manager.py, pool.py, transfer_engine.py
  ├── cache/lfu_cache.py, eviction.py
  └── runtime/engine.py
```

#### Phase 0 → Introspect real model

**[NEW] `introspect/weight_profiler.py`**
- Parse safetensors keys: `model.layers.{L}.block_sparse_moe.experts.{E}.{w1,w2,w3}.weight`
- Output per-expert: `(layer, idx, size_bytes_fp16, size_bytes_int4)`
- Output aggregate: total expert weight vs shared weight breakdown
- Qwen1.5-MoE: 24 layers × 60 experts (4 shared + 56 routed) = 1440 expert blocks

**[NEW] `introspect/routing_tracer.py`**
- `register_forward_hook` on `block_sparse_moe.gate` per layer
- Run ~500-1000 tokens of WikiText through model
- Capture: `(token_idx, layer_idx, top_k_expert_ids, gating_weights)`
- Save as `.npz`
- Also compute: per-expert frequency histogram, basic distribution stats

#### Phase 1 → Core types (minimal)

**[NEW] `core/types.py`**
- `MemoryTier` enum: HBM, DRAM (CXL added in Milestone 2)
- `ExpertId = Tuple[int, int]`
- `TransferState` enum
- `TransferRequest` dataclass

**[NEW] `core/config.py`**
- Single `MemTierConfig` dataclass. No YAML loading yet — just Python defaults with overrides.
- Hardware presets: `RTX4050_PRESET`, `L40S_PRESET` as frozen configs.

```python
@dataclass
class MemTierConfig:
    gpu_vram_bytes: int = 6_442_450_944
    hbm_cache_budget_bytes: int = 4_000_000_000  # reserve 2GB for attn/KV
    host_dram_bytes: int = 16_000_000_000
    pcie_bandwidth_gbps: float = 16.0
    frequency_decay_half_life: int = 500
    eviction_policy: str = "lfu"  # just LFU for M1, size-aware in M2
```

**[NEW] `core/metrics.py`**
- Simple counters dict. No thread safety yet (single-threaded in M1).
- Tracks: cache_hits, cache_misses, evictions, transfers, tokens_processed
- `report() → dict` for printing

#### Phase 2 → Memory engine (2-tier)

**[NEW] `memory/expert_metadata.py`**
- `ExpertMetadata` dataclass initialized from `weight_profiler` output (real sizes)
- Fields: expert_id, size_bytes, current_tier, raw_count, decayed_frequency, last_access_token

**[NEW] `memory/tier_manager.py`**
- `Dict[ExpertId, ExpertMetadata]` registry
- `promote(expert_id)` → DRAM→HBM with capacity check
- `demote(expert_id)` → HBM→DRAM
- Initial placement: top-N experts by frequency (from Phase 0 traces) into HBM, rest in DRAM

**[NEW] `memory/pool.py`**
- `HBMPool`: `torch.cuda` tensors. `store()`, `retrieve()`, `evict()`, `usage_bytes()`
- `DRAMPool`: `torch.Tensor.pin_memory()` on CPU. Same interface.
- No CXL pool yet.

**[NEW] `memory/transfer_engine.py`**
- Single CUDA transfer stream (no double-buffer yet — get correctness first)
- `demand_fetch(expert_id)`: synchronous DRAM→HBM via `tensor.to('cuda', non_blocking=True)` + event wait
- `async_fetch(expert_id)`: same but returns handle, doesn't wait
- Duplicate prevention: skip if already in-flight

#### Phase 3 → LFU cache (simple)

**[NEW] `cache/lfu_cache.py`**
- `dict[ExpertId, CachedExpert]` lookup
- On access: `decayed_freq = old * 0.5^((now - last) / half_life) + 1.0`
- Warm-start from Phase 0 frequency data

**[NEW] `cache/eviction.py`**
- `LFUEviction` only. Evict lowest `decayed_frequency`. Returns `List[ExpertId]` to evict.
- Size-aware and recency variants are Milestone 2 stretch.

#### Phase 5 → Runtime loop (no prefetch)

**[NEW] `runtime/engine.py`**

Simplified 5-step loop (steps 5-6 from README skipped):

```python
def forward_moe_layer(self, layer_idx, hidden_states, router_output):
    expert_ids = router_output.top_k_indices
    hits, misses = self.cache.lookup(expert_ids)

    # Execute hits immediately
    for eid in hits:
        weights = self.cache.get_weights(eid)
        # expert FFN forward

    # Demand-fetch misses (blocking in M1)
    for eid in misses:
        self.transfer_engine.demand_fetch(eid)
        self.cache.insert(eid)  # may trigger eviction
        weights = self.cache.get_weights(eid)
        # expert FFN forward

    self.cache.record_accesses(expert_ids)
    self.metrics.record(hits=len(hits), misses=len(misses))
```

**M1 smoke tests** (cheap insurance — catch math bugs before they corrupt hit-rate numbers 3 milestones later):

```bash
pytest tests/test_weight_profiler.py tests/test_lfu_cache.py tests/test_tier_manager.py -v
```

| Test file | What it covers | Time cost |
|---|---|---|
| `test_weight_profiler.py` | Safetensors key parsing, size calculation against known values | ~2 min to write |
| `test_lfu_cache.py` | Decay math correctness, eviction ordering, capacity enforcement | ~5 min to write |
| `test_tier_manager.py` | Promote/demote, capacity overflow → eviction triggered | ~3 min to write |

These are deliberately minimal — 3-5 assertions each, not a full test suite. The goal is catching a sign error in the decay formula or an off-by-one in capacity checking, not 100% coverage.

**M1 exit criteria**:
- Smoke tests pass
- Run 100 tokens through Qwen1.5-MoE with tiered memory
- Cache hit rate reported
- Model output matches baseline (correctness)
- Demo: show experts moving between DRAM↔HBM in real-time

---

### Milestone 2: CXL Tier + Double-Buffer + Better Eviction [Differentiator]

**Goal**: Add the third tier, async overlap, size-aware eviction. This is the core research contribution.

```
New/modified files (5):
  ├── memory/pool.py              # Add CXLPool
  ├── memory/transfer_engine.py   # Double-buffer, multi-hop CXL→DRAM→HBM
  ├── cache/eviction.py           # Add SizeAwareLFUEviction
  ├── cache/placement.py          # NEW: eviction destination logic
  └── scripts/verify_overlap.py   # NEW: nsys verification script
```

**`memory/pool.py` — add CXLPool**

```python
class CXLPool(MemoryPool):
    """CPU tensors + calibrated busy-wait latency injection."""
    def __init__(self, capacity_bytes, latency_ns=350, bandwidth_gbps=8.0):
        self.latency_ns = latency_ns
        self.token_bucket = TokenBucketRateLimiter(bandwidth_gbps)

    def retrieve(self, expert_id):
        _busy_wait_ns(self.latency_ns)  # NOT time.sleep
        self.token_bucket.acquire(self.experts[expert_id].nbytes)
        return self.experts[expert_id]
```

**`memory/transfer_engine.py` — double-buffer + multi-hop**

- Two ping-pong GPU buffers for overlapping current-expert compute with next-expert transfer
- CXL→HBM is two-hop: CXL→DRAM (CPU memcpy + latency injection), then DRAM→HBM (CUDA async)
- **After implementation**: run `verify_overlap.py` with `nsys` to confirm overlap is real

**[NEW] `cache/placement.py`**

```python
def select_eviction_target(expert: ExpertMetadata, dram_free: int) -> MemoryTier:
    if expert.decayed_frequency > warm_threshold:
        return MemoryTier.DRAM   # likely needed again
    if dram_free > expert.size_bytes:
        return MemoryTier.DRAM   # DRAM has space
    return MemoryTier.CXL        # truly cold
```

Thresholds derived from Phase 0 frequency percentiles (e.g., warm_threshold = p50 of decayed frequencies).

**M2 exit criteria**:
- 3-tier placement working (experts visibly in HBM/DRAM/CXL)
- Profiler trace (nsys if it worked in M0, else torch.profiler chrome trace) shows compute-transfer overlap
- Cache hit rate improves vs M1 (larger effective capacity)
- Benchmark: M1 (2-tier) vs M2 (3-tier) comparison, same workload

---

### Milestone 3: Predictive Prefetcher + Evaluation [The Story]

**Goal**: Add the router-predictive prefetcher and run the full 4-baseline ablation. This is what makes the demo compelling.

```
New files (8):
  ├── introspect/analysis.py             # Co-occurrence extraction
  ├── prefetch/co_occurrence.py          # Cross-layer model
  ├── prefetch/predictor.py              # Confidence-scored predictions
  ├── prefetch/prefetch_scheduler.py     # Queue + bandwidth throttle
  ├── runtime/router_interceptor.py      # Refactored hook for live feeding
  ├── evaluation/baselines.py            # 4 baselines
  ├── evaluation/benchmark_runner.py     # Run matrix
  └── evaluation/visualization.py        # Plots
```

**`prefetch/co_occurrence.py`** — built from real traces (Phase 0)

```python
class CoOccurrenceModel:
    # matrix[src_layer][src_expert][dst_layer][dst_expert] = cond_probability
    # Built from real routing traces, not synthetic
    # Lookahead: 2 layers (enough to hide PCIe latency on RTX 4050)
    # Sparse: only store pairs above 0.05 threshold
```

**`prefetch/predictor.py`** — start simple

- Query co-occurrence model with current layer's routing
- Return predictions above `confidence_threshold` (default 0.3)
- Track precision over rolling window
- **Stretch**: self-tuning threshold. For demo: fixed threshold is fine.

**`prefetch/prefetch_scheduler.py`**

- Filter: skip HBM-resident, skip in-flight
- Sort by confidence
- Submit top-N to transfer_engine (N = `max_inflight_transfers`, default 2-4)
- Record outcomes for precision tracking

**`evaluation/baselines.py`** — 4 configs, same engine

| Baseline | Tiers | Cache | Prefetch |
|---|---|---|---|
| GPU-Resident | HBM only | N/A | No |
| Two-Tier | HBM+DRAM | LRU | No |
| CXL-Only | HBM+DRAM+CXL | LFU | No |
| MemTier-MoE | HBM+DRAM+CXL | LFU | Yes |

**Benchmark matrix (trimmed for demo)**:
- HBM budgets: **3GB, 4GB** (2 configs, not 4)
- Text domains: **WikiText, code** (2 domains, not 3)
- = 4 baselines × 2 budgets × 2 domains = **16 runs**

**M3 exit criteria**:
- Prefetch precision > 50% on stable (WikiText) workload
- MemTier-MoE cache hit rate > CXL-Only (prefetch adds value)
- 4-baseline ablation table with real numbers
- 3-4 plots: hit rate comparison, prefetch precision over time, throughput bars, expert heatmap

---

## Dependency Graph

```mermaid
graph LR
    M0["M0: Env Bootstrap"] --> M1
    M1["M1: 2-Tier PoC<br/>12 files, demoable"] --> M2
    M2["M2: CXL + Overlap<br/>5 files, differentiator"] --> M3
    M3["M3: Prefetcher + Eval<br/>8 files, the story"]

    style M0 fill:#666,color:#fff
    style M1 fill:#4a9eff,color:#fff
    style M2 fill:#ff9f43,color:#fff
    style M3 fill:#51cf66,color:#fff
```

**If time runs out**:
- After M1: demoable 2-tier PoC with real model data. Respectable.
- After M2: 3-tier with verified async overlap. Strong.
- After M3: full story with ablation. Winner-tier.

---

## Hardware Plan

| GPU | Model | Quantization | Fits in VRAM? | Use Case |
|---|---|---|---|---|
| RTX 4050 (6GB) | Qwen1.5-MoE-A2.7B | INT4 (~7.2GB) | ❌ needs offload | **Primary dev** — forces real tiering |
| L40S ×2 (96GB) | Mixtral-8x7B | INT4 (~24GB) | ✅ (constrain artificially) | Scale validation |

---

## Risk Table (updated)

| Risk | Severity | Mitigation |
|---|---|---|
| `auto-gptq` build fails | 🔴 | Try `autoawq` as fallback. Or `bitsandbytes` 4-bit. Test in M0. |
| CUDA stream overlap not real | 🔴 | `nsys` verification in M2. Don't trust without trace proof. |
| Qwen routing too stable → trivially high hit rate | 🟡 | Reduce cache budget to force misses. Use code corpus (different distribution). |
| CXL latency injection inaccurate at ns scale | 🟡 | Busy-wait spin loop. Accepted approximation (~1-3× target, not exact). Relative tier ordering is what matters. |
| `nsys` doesn't work under WSL2 | 🟡 | Test in M0. Fallback: `torch.profiler` + chrome trace export. |
| Co-occurrence overfits to WikiText | 🟡 | Test cross-domain. Online EMA update as stretch. |
| RTX 4050 VRAM too tight for model + cache + KV | 🟡 | Start on L40S, port down. Design is GPU-agnostic. |
| Full benchmark matrix too slow | 🔵 | Trimmed to 16 runs. Can parallelize on L40S. |

---

## Success Criteria

| Metric | Must-have (M1) | Good (M2) | Great (M3) |
|---|---|---|---|
| Working demo | ✅ 2-tier, real model | ✅ 3-tier, async overlap | ✅ + prefetch + ablation |
| Cache hit rate | Reported | > 60% | > 60% with ablation proof |
| Prefetch precision | N/A | N/A | > 50% |
| Throughput vs 2-tier | Measured | Improved (any %) | > 15-20% |
| Correctness | Output matches baseline | Same | Same |

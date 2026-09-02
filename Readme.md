# MemTier-MoE

## CXL-Aware Memory Tiering for Mixture-of-Experts Inference

## Abstract

As Mixture-of-Experts (MoE) large language models scale, GPU High Bandwidth Memory (HBM) capacity becomes a primary constraint during inference. Although models such as Mixtral-8x7B and Qwen2-MoE activate only a small subset of parameters for each token, all expert weights traditionally remain resident in GPU memory to avoid the severe latency of repeated transfers.

Existing offloading systems generally treat memory as a two-tier hierarchy: GPU HBM and host system DRAM. This boundary does not exploit the intermediate capacity, bandwidth, and latency characteristics offered by Compute Express Link (CXL) memory expansion. MemTier-MoE proposes a theoretical framework and runtime engine that makes CXL memory a first-class tier for MoE inference.

The framework combines an LFU-based expert cache with a router-predictive prefetcher. Together, these mechanisms keep frequently used experts in HBM, stage less frequently used experts in host DRAM, and place the coldest weights in a CXL-attached memory pool. The prefetcher uses early routing decisions and learned cross-layer co-occurrence statistics to issue asynchronous DMA transfers before the corresponding experts are needed. The goal is to overlap expert movement with attention computation and reduce the cost of capacity-driven offloading.

## Problem

MoE models increase total parameter capacity without activating every expert for every token. However, inference still faces a residency problem:

- GPU HBM offers excellent bandwidth but limited capacity.
- Host DRAM provides more capacity but incurs higher access and transfer latency.
- CXL memory can expand available memory while offering a distinct performance point between local DRAM and slower storage-backed mechanisms.
- Expert activation is sparse, dynamic, and often correlated across layers, making static placement inefficient.

The central question is:

> Can CXL-aware placement, frequency-based caching, and routing-informed prefetching reduce expert-transfer overhead enough to improve MoE inference throughput under constrained HBM capacity?

## Proposed Architecture

MemTier-MoE exposes a unified three-tier hierarchy to the inference runtime:

| Tier | Memory | Role | Expected access pattern |
| --- | --- | --- | --- |
| Hot | GPU HBM | Resident working set of active and frequently used experts | Lowest latency, highest bandwidth |
| Warm | Host system DRAM | Backup residency for moderately active experts | Higher transfer latency than HBM |
| Cold | CXL-attached memory expansion | Capacity tier for infrequently used experts | Higher latency, large available capacity |

The runtime maintains metadata for each expert, including its current tier, activation frequency, recency, transfer state, size, and observed co-occurrence with other experts. The placement policy is adaptive rather than fixed: experts may move between tiers as the routing distribution changes.

## Core Mechanisms

### 1. LFU-Based Expert Cache

The HBM cache uses an activation-frequency policy to prioritize experts that contribute most often to the current workload. Each routed expert updates a frequency counter, optionally with decay so that historical popularity does not permanently dominate newer routing behavior.

When HBM capacity is insufficient, the cache evicts low-frequency experts to host DRAM or CXL memory according to their current residency and transfer cost. The policy can be extended with recency and size awareness to avoid retaining a large expert at the expense of several smaller, frequently used experts.

### 2. Router-Predictive Prefetcher

The prefetcher observes routing decisions from early layers and predicts experts likely to be requested by later layers. Predictions are based on:

- Current-token and recent-batch routing decisions.
- Cross-layer expert co-occurrence statistics collected during a profiling or warm-up phase.
- Confidence thresholds that control whether a prediction is worth the transfer cost.

For high-confidence predictions, the runtime schedules asynchronous DMA transfers from DRAM or CXL memory to HBM. Transfers use dedicated asynchronous CUDA streams and are coordinated with the compute stream so that expert movement can overlap with attention and other available computation.

Prefetching must remain bounded: incorrect predictions consume bandwidth and may evict useful experts. The engine therefore tracks prediction accuracy, transfer usefulness, and queue pressure to adjust its aggressiveness at runtime.

## Inference Flow

1. The router selects experts for the current token or batch.
2. The cache checks whether the selected experts are already resident in HBM.
3. Cache hits execute immediately on the compute stream.
4. Cache misses trigger promotion from host DRAM or CXL memory.
5. Early routing signals update the prefetcher and its cross-layer prediction state.
6. Prefetched experts are transferred asynchronously while attention computation proceeds.
7. Activation counters, prediction outcomes, and transfer timings update the placement policy.

The intended steady-state behavior is for most expert transfers to complete before the associated expert computation begins, turning memory movement into an overlapped operation rather than a serialized stall.

## Evaluation Methodology

Because physical CXL hardware may not be available, the initial evaluation will use Linux NUMA emulation configured to approximate industry-reported CXL memory characteristics. The emulation should model the relative capacity, bandwidth, and latency differences among GPU HBM, host DRAM, and the CXL tier.

### Baselines

- **GPU-resident baseline:** all expert weights remain in HBM where capacity permits.
- **Two-tier offloading:** expert weights move between GPU HBM and host DRAM without CXL-aware placement.
- **CXL-only tiering:** three-tier placement without predictive prefetching.
- **MemTier-MoE:** LFU expert caching combined with router-predictive prefetching.

### Workloads

The evaluation should cover representative MoE models, batch sizes, sequence lengths, HBM capacity limits, and routing distributions. Both stable routing workloads and distribution-shift workloads are important because they test whether frequency counters and predictions adapt over time.

### Metrics

- Tokens per second and end-to-end latency.
- Expert-cache hit rate and miss rate.
- Prefetch precision, recall, and useful-transfer ratio.
- HBM, DRAM, and CXL bandwidth utilization.
- Per-layer transfer and compute overlap.
- Queueing delay and tail latency.
- Memory footprint and energy implications where measurement is available.

## Target Outcomes

MemTier-MoE targets a 60-85% GPU expert-cache hit rate and higher token throughput than conventional two-tier offloading. The main performance objective is to amortize per-layer expert-transfer latency within the attention-computation window. Results should also identify the workload and capacity conditions under which CXL tiering provides a meaningful advantage.

These values are research targets, not guaranteed results. They should be validated against measured baselines under identical model, hardware, and emulation settings.

## Expected Contributions

1. A three-tier memory abstraction for MoE inference spanning GPU HBM, host DRAM, and CXL memory.
2. An LFU-based expert residency policy adapted to dynamic MoE routing.
3. A router-predictive prefetching strategy using cross-layer expert correlations.
4. An evaluation methodology for studying CXL-aware MoE inference without physical CXL hardware.
5. An analysis of the trade-offs between cache hit rate, prefetch accuracy, bandwidth pressure, and throughput.

## Limitations and Open Questions

- NUMA emulation cannot reproduce every behavior of real CXL devices, including fabric contention and hardware-specific DMA paths.
- LFU counters may react slowly to routing distribution shifts unless decay or windowing is used.
- Incorrect prefetches can increase bandwidth pressure and cause harmful evictions.
- The best tier-placement policy may vary with expert size, batch size, sequence length, and model topology.
- Real deployment requires integration with the model runtime, CUDA memory management, and CXL-capable platform firmware.

## Status

MemTier-MoE is a research design and evaluation framework. The next implementation steps are to define the tier manager and expert metadata format, implement the cache and prefetcher in a simulator or inference runtime, configure NUMA-based experiments, and compare the proposed design with two-tier offloading baselines.

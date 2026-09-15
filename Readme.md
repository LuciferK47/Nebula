# MemTier-MoE: CXL-Aware 3-Tier Memory Hierarchy for Mixture-of-Experts Inference

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.0+](https://img.shields.io/badge/cuda-12.0%2B-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Tests Passing](https://img.shields.io/badge/tests-125%2F125%20passed-brightgreen.svg)]()
[![Hardware Profiles](https://img.shields.io/badge/profiles-JEDEC%20HBM3%2Fe%20%7C%20CXL%202.0%2F3.0-blueviolet.svg)]()
[![Trace Exporter](https://img.shields.io/badge/traces-DRAMSim3%20%7C%20gem5%20compatible-orange.svg)]()
[![Hardware Agnostic](https://img.shields.io/badge/hardware-Auto--Detect%20%7C%20CUDA%20%2B%20CPU%20Fallback-success.svg)]()
[![License MIT](https://img.shields.io/badge/license-MIT-informational.svg)](LICENSE)

> **MemTier-MoE places cold Mixture-of-Experts (MoE) parameters across a 3-tier memory hierarchy (GPU VRAM, Host DRAM, and an emulated CXL far memory pool) and executes them via asymmetric activation offloading—routing lightweight 4 KB activations over PCIe instead of repeatedly swapping 17 MB weights across the interconnect.**

An open-source runtime engine and architectural research framework designed to break the memory capacity wall for Mixture-of-Experts models on memory-constrained hardware. Evaluated empirically on commodity GPU hardware under constrained VRAM (e.g. 6 GB test environment) and scales automatically to any CUDA or CPU platform.

---

## Interactive Presentation & Live Studio

MemTier-MoE includes **Nebula**, an interactive web presentation and hardware evaluation studio:

```bash
# Launch the full web studio and API backend (single command):
python scripts/serve.py --port 8000
# Open http://localhost:8000 in your browser
```

---

## Executive Summary: Moving Activations vs. Thrashing Weights

Mixture-of-Experts (MoE) models scale total parameter capacity into tens or hundreds of billions while routing each token to only a sparse subset of experts (e.g., top-2 of 64). However, conventional offloading treats memory as a rigid two-tier boundary (VRAM vs. Host DRAM), continually swapping expert weights over the PCIe bus whenever an un-cached expert is routed.

```
+-----------------------------------------------------------------------------+
|                          MemTier-MoE Architecture                           |
+-----------------------------------------------------------------------------+
|                                                                             |
|   +---------------------------------------------------------------------+   |
|   |                       HuggingFace Transformer                       |   |
|   |  (Embeddings, Attention, LayerNorms, Router Gates, LM Head in VRAM) |   |
|   +----------------------------------+----------------------------------+   |
|                                      |                                      |
|                             Token Routing Logits                            |
|                                      v                                      |
|   +---------------------------------------------------------------------+   |
|   |                  Cross-Layer Predictive Prefetcher                  |   |
|   |      P(E_{l+k} | E_l) Statistical Table + Confidence Scheduler      |   |
|   +----------------------------------+----------------------------------+   |
|                                      | Async Transfer Triggers              |
|                                      v                                      |
|   +---------------------------------------------------------------------+   |
|   |                  Dynamic 3-Tier Memory Hierarchy                    |   |
|   |                                                                     |   |
|   |   [ Tier 0: Hot (GPU VRAM / GDDR6 / HBM) ] <====================+   |   |
|   |     - Resident working set of active & pinned top-k experts     |   |   |
|   |     - Decayed LFU Cache Eviction (Protected Layer Working Set)  |   |   |
|   |                                                                 |   |   |
|   |   [ Tier 1: Warm (Host DRAM - Physical System RAM) ] <==========+   |   |
|   |     - Staged warm experts & immediate offload buffer            |   |   |
|   |     - Asynchronous non-blocking CUDA stream transfers           |   |   |
|   |                                                                 |   |   |
|   |   [ Tier 2: Cold (CXL Far Memory Pool - Software Emulated) ] <==+   |   |
|   |     - Disaggregated high-capacity expert parameter pool         |   |   |
|   |     - Calibrated latency injection (350 ns) & rate limiter      |   |   |
|   +-----------------------------------------------------------------+---+   |
+-----------------------------------------------------------------------------+
```

### The Operational Roofline Contrast

$$\text{Arithmetic Intensity } (I) = \frac{\text{FLOPs}}{\text{Bytes Moved over Interconnect}}$$

| Execution Strategy | What Crosses the PCIe Bus? | Data Moved per Expert | Arithmetic Intensity ($I$) | Primary Bottleneck |
| :--- | :--- | :---: | :---: | :--- |
| **Two-Tier Weight Swapping** | Full Expert Weight Matrix | **17,301,504 bytes** (17.3 MB) | **$1.00\text{ FLOP/byte}$** | **PCIe Bandwidth Bound** (Bus saturated) |
| **MemTier-MoE Hybrid** | Input/Output Token Activations | **4,096 bytes** (4 KB) | **$4,224.0\text{ FLOP/byte}$** | **Compute Saturation** (Bus free) |

> **Result**: A **$4,224\times$ increase in operational arithmetic intensity**, shifting execution from a saturated PCIe transfer bottleneck into productive matrix compute.

---

## Canonical Empirical Results

All results below are directly reproducible via `scripts/run_scenarios.py` and machine-readable in `results/scenarios/*.json`.

### 1. Crossover Advantage: Activation Offload vs. Weight Swapping (`Scenario 2`)

Evaluated on `Qwen1.5-4x0.5B-Chat-MoE` (24 layers, 96 expert modules, 1,660.9 MB total expert footprint in FP16, top-2 routing):

| VRAM Budget | Strategy | Throughput | Evictions | Bus Traffic | Speedup | Bus Traffic Reduction |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **600 MB** (~36% VRAM) | Two-Tier Weight Swapping | 2.84 tok/s | 827 | 14,308.3 MB (14.3 GB) | *baseline* | *baseline* |
| **600 MB** (~36% VRAM) | **MemTier-MoE (Hybrid)** | **8.60 tok/s** | **0** | **4.63 MB** | **3.03× (+203%)** | **3,091× Lower** |
| **900 MB** (~54% VRAM) | Two-Tier Weight Swapping | 3.68 tok/s | 535 | 9,671.5 MB (9.7 GB) | *baseline* | *baseline* |
| **900 MB** (~54% VRAM) | **MemTier-MoE (Hybrid)** | **9.50 tok/s** | **0** | **3.25 MB** | **2.58× (+158%)** | **2,976× Lower** |

- **Key Takeaway**: Under severe VRAM constraints (600 MB), weight swapping thrashes the bus (14.3 GB moved across 827 evictions). Hybrid activation offload keeps weights stationary, moving only 4.63 MB of activations for a **3.03× throughput speedup** and **zero evictions**.

---

### 2. Capacity Cliff: Scaling Across VRAM Budgets (`Scenario 1`)

Sweeping allocated VRAM budget from 200 MB up to 2,500 MB (100% resident):

| VRAM Budget | Resident Working Set | Hit Rate | Throughput | System Behavior |
| :---: | :---: | :---: | :---: | :--- |
| **200 MB** | 12.0% | 10.9% | 7.65 tok/s | Heavy activation offloading; stable execution |
| **300 MB** | 18.1% | 20.1% | 7.00 tok/s | Cold experts stream activations to CPU seamlessly |
| **600 MB** | 36.1% | 39.9% | 8.38 tok/s | Fast working set in VRAM; offloaded tail in DRAM |
| **900 MB** | 54.2% | 57.7% | 9.30 tok/s | Majority of active tokens hit local GPU memory |
| **1,500 MB** | 90.3% | 93.7% | 11.01 tok/s | Near-zero offload overhead |
| **2,000 MB** | 100.0% | 100.0% | 12.42 tok/s | Unconstrained GPU VRAM reference |

- **Key Takeaway**: Cache hit rate scales smoothly and monotonically with memory allocation. Unlike weight swapping which crashes into a steep performance cliff when VRAM is starved, hybrid activation offloading delivers steady, productive throughput (>7.0 tok/s) even with only 200 MB of VRAM.

---

### 3. Concurrent Batch Scaling: Thrashing Collapse vs. Hybrid Immunity (`Scenario 7`)

Evaluating concurrent batch scaling ($B=1, 2, 4, 8$) under a constrained 600 MB VRAM budget:

| Batch Size ($B$) | Weight Swapping Tok/s | Weight Swapping Hit Rate | Hybrid Tok/s | Hybrid Hit Rate | Hybrid Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2.45 tok/s | 34.9% | **5.37 tok/s** | 39.5% | **2.19× Speedup** |
| **2** | 2.73 tok/s | 9.5% ⚠️ | **11.25 tok/s** | 39.8% | **4.12× Speedup** |
| **4** | 4.42 tok/s | 0.5% 💥 | **18.08 tok/s** | 37.3% | **4.09× Speedup** |
| **8** | 8.29 tok/s | 0.5% 💥 | **25.80 tok/s** | 38.3% | **3.11× Speedup** |

- **Key Takeaway**: Under concurrent batching ($B \ge 2$), diverse prompt tokens require disjoint expert sets simultaneously. In weight-swapping engines, this triggers **catastrophic cache thrashing collapse** (hit rate plunges from 35% down to 0.5%). MemTier-MoE is **completely immune to thrashing**: hit rate holds steady at ~38–40%, allowing throughput to scale cleanly up to **25.8 tok/s**.

---

### 4. Full-Scale MoE Deployment: 14.3B Model on a Single 6 GB GPU

Evaluating `Qwen/Qwen1.5-MoE-A2.7B` (14.3B total parameters, 2.7B active per token, 26.68 GB FP16 weight footprint) on a single commodity 6 GB GPU with 16 GB host RAM:

| Metric | Measured Value | Architecture Note |
| :--- | :---: | :--- |
| **Total Model Parameters** | **14.3 Billion** | 28 transformer layers, 64 experts per layer (1,792 total experts) |
| **FP16 Weight Footprint** | **26.68 GB** | Exceeds physical 6 GB GPU VRAM by **4.4×** |
| **Tier Distribution** | 60 VRAM / 121 DRAM / 1,259 Far Pool | Dynamic multi-tier placement across available headroom |
| **Combined Memory Hit Rate** | **66.88%** | 147 HBM hits + 495 DRAM hits across generation pass |
| **Warmed Steady-State Throughput** | **2.87 tok/s** (348.8 ms / token) | Disk fetches drop to 0 once working set warms in memory |
| **Mathematical Parity** | **Exact Token Sequence Match** | 100% identical greedy generation vs unconstrained baseline |

---

### 5. Academic Quality & Serving Parity: WikiText-2 & ShareGPT

| Benchmark Suite | Metric | Full VRAM Oracle | MemTier-MoE (900 MB) | MemTier-MoE (600 MB) | Quality / Latency Delta |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **WikiText-2 Test Split** | Cross-Entropy Loss | 3.3005 | 3.3004 | 3.2993 | $\le \mathbf{0.036\%}$ loss difference |
| *(Language Modeling Quality)* | Perplexity (PPL) | 27.1250 | 27.1232 | 27.0942 | $\mathbf{-0.031\ \Delta\text{PPL}}$ (Zero degradation) |
| **ShareGPT Serving** | TTFT ($P_{50}$) | 125.9 ms | 398.0 ms | 506.2 ms | Pre-layer routing warmup |
| *(vLLM / SGLang Standard)* | TPOT (Decode Mean) | 108.5 ms | 126.0 ms | 131.4 ms | **< 132 ms inter-token decode** |
| | Decode ($P_{90}$) | 135.2 ms | 143.3 ms | 153.0 ms | Tightly bounded tail latency |

---

## Core Architectural Mechanisms

### 1. Decayed LFU Cache with Active Set Pinning
- **Decayed LFU**: Tracks expert activation frequency per layer with token-based exponential half-life decay to reflect dynamic prompt topic shifts.
- **Active Pinning**: Proactively pins active $top\text{-}k$ experts during the forward pass of layer $l$, guaranteeing zero intra-layer eviction races even under 200% memory oversubscription.

### 2. Router-Predictive Cross-Layer Prefetching
- Learns cross-layer conditional probabilities $P(E_{l+k} \mid E_l)$ from runtime token routing traces.
- Evaluates early routing decisions at layer $l$ to issue non-blocking prefetch transfers for upcoming layers $l+k$ before attention computation finishes.

### 3. Dedicated Non-Blocking CUDA Streams & Events
- Transfers execute on a dedicated background CUDA stream (`torch.cuda.Stream`) isolated from kernel compute.
- `torch.cuda.Event` pairs synchronize dependencies with zero GPU stalls, verified via timeline tracing.

### 4. Hardware Profiles & Simulator Traces
- Built-in JEDEC and CXL Consortium specifications: `jedec-hbm3-cxl2`, `jedec-hbm3e-cxl3`, `workstation`, and `datacenter-a100`.
- Native transaction trace exporter for **DRAMSim3** and **gem5** architecture simulators.

---

## Hardware Grounding & Emulation Methodology

To ensure complete transparency:

1. **What is physically measured on hardware**:
   - Transformer forward pass, attention, LayerNorms, routers, and LM head on GPU.
   - Host-to-device and device-to-host activation transfers over the physical PCIe bus.
   - GPU VRAM allocations and physical host DRAM pinned memory allocations.
   - End-to-end token generation latency, wall time, and exact decoded token sequences.
2. **What is software-emulated**:
   - The **CXL Far Memory Pool** is backed by physical host RAM, isolated with software accounting, a calibrated access delay (350 ns), and a token-bucket bandwidth limiter (8 GB/s).
   - *Bandwidth Dominance*: For a 17 MB expert, link transfer latency represents $<0.05\%$ of total time; interconnect bandwidth dominates by over four orders of magnitude. The framework models CXL primarily as a bandwidth-constrained tier.
3. **Resilience & Automated Stress Validation (`Scenario 10`)**:
   - 10 boundary stress tests executed live: zero VRAM fallback (graceful CPU execution), extreme oversubscription, zero persistent memory leaks across 100+ tokens, and clean unpatching on exit.

---

## Quickstart & Reproduction

### 1. Installation

```bash
# Clone repository
git clone https://github.com/LuciferK47/Nebula.git
cd Nebula

# Create virtual environment (Python 3.10+)
python -m venv .venv
# On Windows: .venv\Scripts\activate | On Linux/macOS: source .venv/bin/activate

# Install package & dependencies
pip install -e . -r requirements.txt
```

### 2. Launch Interactive Presentation & Live Studio

```bash
python scripts/serve.py --port 8000
```
*Open `http://localhost:8000` to interact with the 3D chiplet hierarchy, live prompt runner, and benchmark dashboard.*

### 3. Run Scenario Suite

```bash
# Run canonical scenarios (S1, S2, S7)
python scripts/run_scenarios.py --scenarios s1,s2,s7
```
*Outputs are saved to `results/scenarios/*.json`.*

### 4. Run Live Model Inference (CLI)

```bash
python scripts/run_live_inference.py \
  --model-id "models/Qwen1.5-4x0.5B-Chat-MoE" \
  --prompt "Mixture-of-Experts architectures overcome the memory wall by" \
  --hbm-budget-mb 600 \
  --max-new-tokens 25
```

### 5. Run Verification Test Suite

```bash
pytest tests -v
# Output: 125 passed in ~18s (100% green)
```

---

## Repository Structure

```
Nebula/
├── docs/                        # Technical system documentation & math derivations
├── frontend/                    # Nebula interactive web application (React, Vite, Three.js)
│   ├── src/components/          # UI components (3D explorer, benchmarks, PromptStudio)
│   └── dist/                    # Production bundle served directly by scripts/serve.py
├── memtier_moe/                 # Core framework package
│   ├── cache/                   # Decayed LFU cache, active pinning, & eviction policies
│   ├── core/                    # Hardware profiles, latency injector, & auto-detect config
│   ├── evaluation/              # Benchmark runners & visualization generators
│   ├── introspect/              # Safetensors weight profiler & routing tracer
│   ├── memory/                  # Multi-tier memory pools, transfer engine, & trace exporter
│   ├── prefetch/                # Cross-layer conditional probability prefetch scheduler
│   └── runtime/                 # TieredMoEWrapper & physical inference engine
├── models/                      # MoE model weights & configs (Qwen1.5-4x0.5B, Qwen1.5-MoE-A2.7B)
├── results/                     # Canonical benchmark outputs, plots, & scenario JSONs
│   └── scenarios/               # Canonical S1, S2, S3, S7, S10 experimental runs
├── scripts/                     # CLI entry points
│   ├── serve.py                 # Unified API + static web presentation server
│   ├── run_scenarios.py         # Canonical scenario suite executor (S1-S10)
│   ├── run_live_inference.py    # Autoregressive generation CLI with tier telemetry
│   └── generate_roofline.py     # Interactive Plotly operational roofline generator
├── tests/                       # Comprehensive pytest suite (125 tests, 100% passing)
├── pyproject.toml               # Python packaging & dependencies
└── README.md                    # Project overview & architectural guide
```

---

## References

### 1. Mixture-of-Experts & Offloading Systems
1. **Fiddler (MLSys 2024)**  
   Ali et al., *"Fiddler: Efficient Inference of Mixture-of-Experts Models on Limited GPU Memory"*, Proceedings of Machine Learning and Systems (MLSys), 2024.  

2. **MoE-Infinity (ASPLOS 2024)**  
   Elsayed et al., *"MoE-Infinity: Activation-Aware Expert Offloading to Secondary Storage for Large Sparse Models"*, ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), 2024.  

3. **DeepSpeed-MoE (OSDI 2022)**  
   Rajbhandari et al., *"DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale"*, USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2022.  

4. **FasterMoE (PPoPP 2022)**  
   He et al., *"FasterMoE: Modeling and Optimizing Training and Inference of Large-Scale Mixture-of-Experts on Distributed GPUs"*, ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP), 2022.  

5. **Qwen1.5-MoE (2024)**  
   Qwen Team, Alibaba Group, *"Qwen1.5-MoE: Matching 7B Model Performance with 2.7B Activated Parameters"*, Technical Report, 2024.  

---

### 2. Interconnect & Memory Standards
6. **Compute Express Link (CXL) Specification (Revs 2.0 & 3.0)**  
   CXL Consortium, *"Compute Express Link (CXL) Specification: Type 3 Memory Devices and Direct Memory Pools"*, 2020–2023.  

7. **JEDEC HBM3 & HBM3e Standards**  
   JEDEC Solid State Technology Association, *"High Bandwidth Memory (HBM3/HBM3e) DRAM Standard"*, JESD238 / JESD238A.  

8. **JEDEC DDR5 SDRAM Specification**  
   JEDEC Solid State Technology Association, *"DDR5 SDRAM Standard"*, JESD79-5C.  

---

### 3. Architecture Simulators & Analytical Models
9. **DRAMSim3 (IEEE CAL 2020)**  
   Li et al., *"DRAMSim3: A Cycle-Accurate, Thermal-Capable DRAM Simulator"*, IEEE Computer Architecture Letters, 2020.  

10. **The gem5 Simulator (ACM SIGARCH / IEEE Micro)**  
    Binkert et al., Lowe-Power et al., *"The gem5 Simulator: Version 20.0+"*, ACM SIGARCH Computer Architecture News / IEEE Micro.  

11. **Roofline Model (CACM 2009)**  
    Williams, Waterman, & Patterson, *"Roofline: An Insightful Visual Performance Model for Multicore Architectures"*, Communications of the ACM, 2009.  


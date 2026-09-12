"""Benchmark runner: executes the evaluation matrix across baselines.

Runs each baseline configuration against routing decision traces,
collects metrics, and produces a structured results table.

Trimmed matrix for demo:
  4 baselines × 2 HBM budgets × 2 text domains = 16 runs
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from memtier_moe.core.config import MemTierConfig
from memtier_moe.core.metrics import MetricsTracker
from memtier_moe.core.types import ExpertId, MemoryTier
from memtier_moe.memory.tier_manager import TierManager
from memtier_moe.memory.transfer_engine import TransferEngine
from memtier_moe.cache.lfu_cache import LFUExpertCache
from memtier_moe.prefetch.co_occurrence import CoOccurrenceModel
from memtier_moe.prefetch.predictor import ExpertPredictor
from memtier_moe.prefetch.prefetch_scheduler import PrefetchScheduler
from memtier_moe.runtime.engine import InferenceEngine
from memtier_moe.evaluation.baselines import BaselineConfig, BaselineType

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    baseline_name: str
    baseline_type: str
    hbm_budget_gb: float
    domain: str
    num_tokens: int
    cache_hit_rate: float
    cache_hits: int
    cache_misses: int
    evictions: int
    prefetch_precision: float
    prefetch_total: int
    total_transfer_bytes: int
    total_transfer_time_ms: float
    wall_time_seconds: float
    tokens_per_second: float  # Modeled/simulated pipeline throughput
    simulation_wallclock_tokens_per_second: float = 0.0  # Python simulation loop throughput (not real model inference)
    measured_tokens_per_second: float = 0.0  # Kept as alias for backward compatibility

    # First-class hierarchical memory tier & CXL expansion metrics
    hbm_hit_rate: float = 0.0
    dram_hit_rate: float = 0.0
    cxl_hit_rate: float = 0.0
    disk_fault_rate: float = 0.0
    hbm_hits: int = 0
    dram_hits: int = 0
    cxl_hits: int = 0
    disk_faults: int = 0
    msr: float = 1.0
    dsar_pct: float = 100.0
    amat_ns: float = 28.0
    cxl_transfer_mb: float = 0.0
    dram_budget_gb: float = 2.0
    cxl_capacity_gb: float = 32.0
    cxl_usage_gb: float = 0.0

    # Methodology metadata: explicit classification of how each metric was produced
    benchmark_mode: str = "simulation"  # "simulation" or "live"
    computation_type: str = "modeled"  # "modeled" (calibrated analytical latency) or "measured" (real forward pass)
    transfer_latency_type: str = "modeled"  # "modeled" (calibrated latency/bandwidth) or "measured" (CUDA events)
    cxl_type: str = "emulated"  # "emulated" (host RAM with delay/rate limiting) or "physical"
    hardware_device: str = "NVIDIA GeForce RTX 4050 Laptop GPU (6GB GDDR6)"

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkMatrix:
    """Full benchmark matrix results."""
    results: List[BenchmarkResult] = field(default_factory=list)

    def to_table(self) -> str:
        """Format results as a rich CXL-aware memory tiering text table."""
        if not self.results:
            return "No results."

        header = (
            f"{'Baseline':<24} {'Domain':<8} {'HBM(GB)':<7} "
            f"{'H_HBM':<7} {'H_DRAM':<7} {'H_CXL':<7} {'H_Disk':<7} "
            f"{'AMAT(ns)':<9} {'DSAR(%)':<8} {'SimWallTok/s':<13} {'ModPipeTok/s':<13} {'Wall(s)':<7}"
        )
        lines = [header, "-" * len(header)]

        for r in self.results:
            line = (
                f"{r.baseline_name:<24} {r.domain:<8} "
                f"{r.hbm_budget_gb:<7.1f} "
                f"{r.hbm_hit_rate:<7.1%} {r.dram_hit_rate:<7.1%} "
                f"{r.cxl_hit_rate:<7.1%} {r.disk_fault_rate:<7.1%} "
                f"{r.amat_ns:<9.1f} {r.dsar_pct:<8.1f} "
                f"{r.simulation_wallclock_tokens_per_second:<13.1f} "
                f"{r.tokens_per_second:<13.1f} "
                f"{r.wall_time_seconds:<7.2f}"
            )
            lines.append(line)

        return "\n".join(lines)

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Convert results to list of dicts for plotting / pandas."""
        return [
            {
                "baseline": r.baseline_name,
                "baseline_type": r.baseline_type,
                "domain": r.domain,
                "hbm_budget_gb": r.hbm_budget_gb,
                "hit_rate": r.cache_hit_rate,
                "hbm_hit_rate": r.hbm_hit_rate,
                "dram_hit_rate": r.dram_hit_rate,
                "cxl_hit_rate": r.cxl_hit_rate,
                "disk_fault_rate": r.disk_fault_rate,
                "hits": r.cache_hits,
                "hbm_hits": r.hbm_hits,
                "dram_hits": r.dram_hits,
                "cxl_hits": r.cxl_hits,
                "disk_faults": r.disk_faults,
                "msr": r.msr,
                "dsar_pct": r.dsar_pct,
                "amat_ns": r.amat_ns,
                "cxl_transfer_mb": r.cxl_transfer_mb,
                "dram_budget_gb": r.dram_budget_gb,
                "cxl_capacity_gb": r.cxl_capacity_gb,
                "cxl_usage_gb": r.cxl_usage_gb,
                "misses": r.cache_misses,
                "evictions": r.evictions,
                "prefetch_precision": r.prefetch_precision,
                "simulation_wallclock_tokens_per_second": r.simulation_wallclock_tokens_per_second,
                "measured_tokens_per_second": r.measured_tokens_per_second,
                "tokens_per_second": r.tokens_per_second,
                "simulated_pipeline_tokens_per_second": r.tokens_per_second,
                "wall_time_s": r.wall_time_seconds,
                "benchmark_mode": r.benchmark_mode,
                "computation_type": r.computation_type,
                "transfer_latency_type": r.transfer_latency_type,
                "cxl_type": r.cxl_type,
                "hardware_device": r.hardware_device,
            }
            for r in self.results
        ]



class BenchmarkRunner:
    """Runs the evaluation matrix.

    Parameters
    ----------
    expert_sizes : dict[ExpertId, int]
        Expert ID → size in bytes (from weight profiler).
    initial_frequencies : dict[ExpertId, float], optional
        Expert ID → frequency (from routing tracer).
    co_occurrence_model : CoOccurrenceModel, optional
        Trained co-occurrence model for prefetch baselines.
    """

    def __init__(
        self,
        expert_sizes: Dict[ExpertId, int],
        initial_frequencies: Optional[Dict[ExpertId, float]] = None,
        co_occurrence_model: Optional[CoOccurrenceModel] = None,
    ) -> None:
        self.expert_sizes = expert_sizes
        self.initial_frequencies = initial_frequencies or {}
        self.co_occurrence_model = co_occurrence_model

    def run_single(
        self,
        baseline: BaselineConfig,
        routing_decisions: List[List[List[int]]],
        domain: str = "unknown",
    ) -> BenchmarkResult:
        """Run a single baseline on the given routing decisions.

        Parameters
        ----------
        baseline : BaselineConfig
            The baseline configuration to use.
        routing_decisions : list[list[list[int]]]
            Shape: [num_tokens][num_layers][top_k_experts]
        domain : str
            Label for the text domain (e.g., "wikitext", "code").

        Returns
        -------
        BenchmarkResult with all metrics.
        """
        config = baseline.config
        num_tokens = len(routing_decisions)

        # Build engine
        engine = InferenceEngine(config)
        engine.setup_from_profile(self.expert_sizes, self.initial_frequencies)

        # Build prefetch pipeline if enabled
        scheduler = None
        if baseline.enable_prefetch and self.co_occurrence_model is not None:
            predictor = ExpertPredictor(
                model=self.co_occurrence_model,
                confidence_threshold=config.prefetch_confidence_threshold,
            )
            scheduler = PrefetchScheduler(
                predictor=predictor,
                transfer_engine=engine.transfer_engine,
                tier_manager=engine.tier_manager,
                cache=engine.cache,
                config=config,
                metrics=engine.metrics,
            )

        # Run
        wall_start = time.perf_counter()

        for token_idx in range(num_tokens):
            engine.metrics.record_token()
            for layer_idx, experts in enumerate(routing_decisions[token_idx]):
                # Drain transfers completed since previous layer so prefetched
                # experts become resident in cache before this layer accesses them
                if scheduler is not None:
                    scheduler.poll_and_complete()

                # Feed prefetcher before processing
                if scheduler is not None:
                    pinned_set = {(layer_idx, exp_idx) for exp_idx in experts}
                    scheduler.on_routing_decision(layer_idx, experts, pinned=pinned_set)

                # Process layer
                engine.forward_moe_layer(layer_idx, None, experts)

                # Mark accessed for precision tracking
                if scheduler is not None:
                    for exp_idx in experts:
                        scheduler.on_expert_accessed((layer_idx, exp_idx))

            # Between tokens: final drain of any remaining completions
            if scheduler is not None:
                scheduler.poll_and_complete()

        wall_time = time.perf_counter() - wall_start
        report = engine.metrics.report()

        prefetch_prec = 0.0
        prefetch_total = 0
        if scheduler is not None:
            stats = scheduler.stats()
            prefetch_prec = stats.get("prefetch_precision", 0.0)
            prefetch_total = int(stats.get("prefetch_total", 0))

        # Hardware-calibrated pipeline simulation:
        # Base GPU computation latency calibrated to physical RTX 4050:
        #   - For Qwen1.5-4x0.5B (0.5B active params): ~85 ms per token = ~11.8 tok/s
        #   - For Qwen1.5-MoE-A2.7B (2.7B active params): ~430 ms per token = ~2.32 tok/s
        num_layers = len(routing_decisions[0]) if num_tokens > 0 and len(routing_decisions) > 0 else 24
        max_exp_id = max((e[1] for e in self.expert_sizes.keys()), default=0)
        if max_exp_id > 8:
            t_compute_token_s = 0.430  # Calibrated to physical RTX 4050 on Qwen1.5-MoE-A2.7B (2.32 tok/s)
        else:
            t_compute_token_s = 0.085  # Calibrated to physical RTX 4050 on Qwen1.5-4x0.5B (11.76 tok/s)

        t_layer_compute_s = t_compute_token_s / max(num_layers, 1)
        total_compute_time_s = num_tokens * t_compute_token_s

        misses = report.get("cache_misses", 0)
        total_transfer_time_s = report.get("total_transfer_time_ms", 0.0) / 1000.0

        if baseline.baseline_type == BaselineType.GPU_RESIDENT:
            pipeline_stall_s = 0.0
        elif not baseline.enable_prefetch:
            # Reactive offloading: every demand miss stalls the execution pipeline
            pipeline_stall_s = total_transfer_time_s
        else:
            # MemTier-MoE: Prefetches overlap DMA transfers with GPU compute
            lookahead = getattr(config, "prefetch_lookahead_layers", 2)
            overlap_window_per_prefetch_s = lookahead * t_layer_compute_s
            useful_prefetches = report.get("prefetch_useful", 0)
            total_transfers = report.get("prefetch_issued", 0) + misses
            avg_xfer_per_expert_s = (total_transfer_time_s / max(total_transfers, 1))

            # Prefetches that completed in time hide their latency behind compute
            unhidden_prefetch_stall_s = max(0.0, avg_xfer_per_expert_s - overlap_window_per_prefetch_s) * useful_prefetches
            demand_miss_stall_s = misses * avg_xfer_per_expert_s
            pipeline_stall_s = demand_miss_stall_s + unhidden_prefetch_stall_s

        simulated_pipeline_time_s = total_compute_time_s + pipeline_stall_s
        simulated_tok_per_sec = num_tokens / simulated_pipeline_time_s if simulated_pipeline_time_s > 0 else 0.0

        sim_wall_tok_s = num_tokens / wall_time if wall_time > 0 else 0.0

        return BenchmarkResult(
            baseline_name=baseline.name,
            baseline_type=baseline.baseline_type.value,
            hbm_budget_gb=config.hbm_cache_budget_bytes / 1e9,
            domain=domain,
            num_tokens=num_tokens,
            cache_hit_rate=report.get("hit_rate", 0.0),
            cache_hits=report.get("cache_hits", 0),
            cache_misses=report.get("cache_misses", 0),
            evictions=report.get("evictions", 0),
            prefetch_precision=prefetch_prec,
            prefetch_total=prefetch_total,
            total_transfer_bytes=report.get("total_transfer_bytes", 0),
            total_transfer_time_ms=report.get("total_transfer_time_ms", 0.0),
            wall_time_seconds=wall_time,
            tokens_per_second=simulated_tok_per_sec,
            simulation_wallclock_tokens_per_second=sim_wall_tok_s,
            measured_tokens_per_second=sim_wall_tok_s,
            hbm_hit_rate=report.get("hbm_hit_rate", report.get("hit_rate", 0.0)),
            dram_hit_rate=report.get("dram_hit_rate", 0.0),
            cxl_hit_rate=report.get("cxl_hit_rate", 0.0),
            disk_fault_rate=report.get("disk_fault_rate", 0.0),
            hbm_hits=report.get("hbm_hits", report.get("cache_hits", 0)),
            dram_hits=report.get("dram_hits", 0),
            cxl_hits=report.get("cxl_hits", 0),
            disk_faults=report.get("disk_faults", 0),
            msr=report.get("memory_service_rate", 1.0),
            dsar_pct=report.get("dsar_pct", 100.0),
            amat_ns=report.get("amat_ns", 28.0),
            cxl_transfer_mb=report.get("cxl_transfer_mb", 0.0),
            dram_budget_gb=config.host_dram_bytes / 1e9,
            cxl_capacity_gb=config.cxl_memory_bytes / 1e9,
            cxl_usage_gb=round(report.get("tier_summary", {}).get("cxl", {}).get("usage_bytes", 0) / 1e9, 2) if config.cxl_memory_bytes > 0 else 0.0,
            benchmark_mode="simulation",
            computation_type="modeled",
            transfer_latency_type="modeled",
            cxl_type="emulated",
            hardware_device="NVIDIA GeForce RTX 4050 Laptop GPU (6GB GDDR6)",
            extra={
                "simulated_pipeline_time_s": simulated_pipeline_time_s,
                "python_wall_time_s": wall_time,
            },
        )

    def run_matrix(
        self,
        baselines: Dict[BaselineType, BaselineConfig],
        routing_data: Dict[str, List[List[List[int]]]],
    ) -> BenchmarkMatrix:
        """Run the full benchmark matrix.

        Parameters
        ----------
        baselines : dict
            Mapping from BaselineType → BaselineConfig.
        routing_data : dict
            Mapping from domain name → routing_decisions.

        Returns
        -------
        BenchmarkMatrix with all results.
        """
        matrix = BenchmarkMatrix()

        for bt, baseline in baselines.items():
            for domain, decisions in routing_data.items():
                logger.info(
                    f"Running: {baseline.name} on {domain} "
                    f"({len(decisions)} tokens)"
                )
                result = self.run_single(baseline, decisions, domain)
                matrix.results.append(result)
                logger.info(
                    f"  → hit_rate={result.cache_hit_rate:.3f}, "
                    f"tok/s={result.tokens_per_second:.1f}"
                )

        return matrix

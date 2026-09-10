"""Engine for transferring experts between memory tiers.

M2 upgrade:
  - Double-buffer (ping-pong) GPU staging for compute-transfer overlap.
  - Multi-hop CXL → DRAM → HBM transfers.
  - Transfer timing recorded per-hop for profiling.
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from memtier_moe.core.config import MemTierConfig
from memtier_moe.core.metrics import MetricsTracker
from memtier_moe.core.types import ExpertId, MemoryTier, TransferRequest, TransferState
from memtier_moe.memory.tier_manager import TierManager

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


@dataclass
class TransferHandle:
    """Handle for an in-flight transfer."""
    request: TransferRequest
    event: Optional[Any] = None       # torch.cuda.Event when applicable
    start_time: float = 0.0
    buffer_idx: int = 0               # which ping-pong buffer (0 or 1)
    hops_completed: int = 0           # for multi-hop CXL→DRAM→HBM
    total_hops: int = 1


@dataclass
class HopTiming:
    """Timing for a single hop in a multi-hop transfer."""
    source_tier: MemoryTier
    dest_tier: MemoryTier
    duration_ms: float


# ── Double-buffer manager ──────────────────────────────────────────────


class DoubleBuffer:
    """Manages two ping-pong GPU staging buffers for overlapping transfers.

    While buffer A holds the expert currently being computed on, buffer B
    receives the next expert from a lower tier.  On each swap the roles
    reverse, hiding transfer latency behind computation.
    """

    def __init__(self, buffer_size_bytes: int) -> None:
        self.buffer_size = buffer_size_bytes
        self._current: int = 0  # index of the *compute* buffer
        self._buffers: List[Optional[Any]] = [None, None]
        self._expert_ids: List[Optional[ExpertId]] = [None, None]

    @property
    def compute_idx(self) -> int:
        """Index of the buffer being read for computation."""
        return self._current

    @property
    def transfer_idx(self) -> int:
        """Index of the buffer being written to by a transfer."""
        return 1 - self._current

    def swap(self) -> None:
        """Swap compute ↔ transfer roles."""
        self._current = 1 - self._current

    def set_tensor(self, idx: int, expert_id: ExpertId, tensor: Any) -> None:
        self._buffers[idx] = tensor
        self._expert_ids[idx] = expert_id

    def get_tensor(self, idx: int) -> Optional[Any]:
        return self._buffers[idx]

    def get_expert(self, idx: int) -> Optional[ExpertId]:
        return self._expert_ids[idx]

    def clear(self, idx: int) -> None:
        self._buffers[idx] = None
        self._expert_ids[idx] = None


# ── Transfer engine ────────────────────────────────────────────────────


class TransferEngine:
    """Manages the transfer of experts between memory tiers.

    M2 features:
      - Multi-hop: CXL→DRAM (CPU memcpy + latency injection), then DRAM→HBM
        (CUDA async).
      - Double-buffer: two staging buffers for overlapping compute with the
        next expert transfer.
      - Dedicated CUDA transfer stream separate from the compute stream.
    """

    def __init__(
        self,
        tier_manager: TierManager,
        config: MemTierConfig,
        metrics: MetricsTracker,
        ensure_room_fn: Optional[Callable[[int], List[ExpertId]]] = None,
    ):
        """
        Args:
            ensure_room_fn: Optional callback, typically
                ``LFUExpertCache.make_room``, invoked with the incoming
                expert's size right before it's stored into HBM in
                ``wait_for``. An async transfer can reserve HBM space at
                issue time (the prefetch scheduler does), but completion
                happens later in the background — an intervening
                demand-fetch can claim that space in between, so the store
                needs its own fresh room check rather than trusting a
                reservation made who-knows-how-long ago. Standalone tests
                that construct a bare TransferEngine leave this ``None``
                and get the old unconditional-store behavior.
        """
        self.tier_manager = tier_manager
        self.config = config
        self.metrics = metrics
        self._ensure_room_fn = ensure_room_fn

        # CUDA transfer stream (separate from default compute stream)
        self._transfer_stream = None
        if HAS_TORCH and torch.cuda.is_available():
            self._transfer_stream = torch.cuda.Stream()

        self._inflight: Dict[ExpertId, TransferHandle] = {}
        self._completed: Set[ExpertId] = set()

        # Double-buffer: exposed for callers that want the classic 2-slot
        # ping-pong view of "the buffer being computed on" vs "the buffer
        # receiving the next transfer" (see the `double_buffer` property).
        # It is NOT used to stage in-flight tensors below — config allows
        # up to `max_inflight_transfers` (default 4) concurrent async
        # fetches, and a fixed 2-slot round-robin index collides once more
        # than 2 are in flight: a 3rd/4th async_fetch would silently
        # overwrite or clear a still-pending transfer's tensor. Staging is
        # keyed by expert_id instead, which has no such capacity limit.
        self._double_buffer = DoubleBuffer(buffer_size_bytes=64 * 1024 * 1024)
        self._next_buffer: int = 0  # cosmetic ping-pong role, see buffer_idx below
        self._staged_tensors: Dict[ExpertId, Any] = {}

    # ── Public API ─────────────────────────────────────────────────────

    def demand_fetch(self, expert_id: ExpertId) -> Any:
        """Synchronously fetch an expert to HBM, using multi-hop if needed."""
        start_time = time.perf_counter()

        metadata = self.tier_manager.get_metadata(expert_id)
        if metadata.current_tier == MemoryTier.HBM:
            return self.tier_manager.get_pool(MemoryTier.HBM).retrieve(expert_id)

        request = self._create_transfer_request(expert_id)
        hop_timings: List[HopTiming] = []

        # Multi-hop: walk the expert up through each tier
        tensor = self._execute_multihop(expert_id, request, hop_timings)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.metrics.record_transfer(request.size_bytes, elapsed_ms)

        logger.debug(
            f"demand_fetch {expert_id}: {len(hop_timings)} hop(s), "
            f"{elapsed_ms:.2f}ms total"
        )
        return tensor

    def async_fetch(self, expert_id: ExpertId) -> TransferHandle:
        """Issue an asynchronous transfer for an expert to HBM.

        Uses the double-buffer staging area and the dedicated transfer
        stream.  Returns immediately with a handle; call wait_for() to
        collect the result.
        """
        if self.is_inflight(expert_id):
            return self._inflight[expert_id]

        metadata = self.tier_manager.get_metadata(expert_id)
        request = self._create_transfer_request(expert_id)
        metadata.transfer_state = TransferState.IN_FLIGHT

        # Determine hop count
        hops = self._hop_count(metadata.current_tier, MemoryTier.HBM)
        buf_idx = self._next_buffer
        self._next_buffer = 1 - self._next_buffer  # alternate

        start_time = time.perf_counter()

        # --- Hop 1 (CPU-side): CXL→DRAM if starting from CXL -----------
        if metadata.current_tier == MemoryTier.CXL:
            source_pool = self.tier_manager.get_pool(MemoryTier.CXL)
            tensor = source_pool.retrieve(expert_id)   # latency injected
            source_pool.evict(expert_id)
            dram_pool = self.tier_manager.get_pool(MemoryTier.DRAM)
            dram_pool.store(expert_id, tensor)
            metadata.current_tier = MemoryTier.DRAM

        # --- Hop 2 (GPU-side): DRAM→HBM --------------------------------
        source_pool = self.tier_manager.get_pool(MemoryTier.DRAM)
        tensor = source_pool.retrieve(expert_id)

        event = None
        if HAS_TORCH and torch.cuda.is_available() and self._transfer_stream is not None:
            with torch.cuda.stream(self._transfer_stream):
                if isinstance(tensor, torch.nn.Module):
                    for p in tensor.parameters():
                        p.data = p.data.to(device="cuda", non_blocking=True)
                    for b in tensor.buffers():
                        b.data = b.data.to(device="cuda", non_blocking=True)
                elif isinstance(tensor, torch.Tensor):
                    tensor = tensor.cuda(non_blocking=True)
                elif hasattr(tensor, "cuda"):
                    try:
                        tensor = tensor.cuda(non_blocking=True)
                    except TypeError:
                        tensor = tensor.cuda()
                event = torch.cuda.Event()
                event.record(self._transfer_stream)

        # Stage the tensor for wait_for()/poll_completed() to pick up. Keyed
        # by expert_id (see __init__ note on why the 2-slot double_buffer
        # itself isn't used here).
        self._staged_tensors[expert_id] = tensor

        handle = TransferHandle(
            request=request,
            event=event,
            start_time=start_time,
            buffer_idx=buf_idx,
            hops_completed=0,
            total_hops=hops,
        )
        self._inflight[expert_id] = handle
        logger.debug(f"async_fetch {expert_id}: issued ({hops} hop(s), buf {buf_idx})")
        return handle

    def wait_for(self, expert_id: ExpertId) -> Optional[Any]:
        """Block until the async transfer for *expert_id* completes."""
        if expert_id not in self._inflight:
            if expert_id in self._completed:
                self._completed.discard(expert_id)
                pool = self.tier_manager.get_pool(MemoryTier.HBM)
                return pool.retrieve(expert_id) if pool.contains(expert_id) else None
            return None

        handle = self._inflight[expert_id]

        # Wait on CUDA event
        if handle.event is not None and HAS_TORCH and torch.cuda.is_available():
            handle.event.synchronize()

        # Finalize: evict from DRAM, promote into HBM pool
        tensor = self._staged_tensors.pop(expert_id, None)

        source_pool = self.tier_manager.get_pool(MemoryTier.DRAM)
        source_pool.evict(expert_id)

        metadata = self.tier_manager.get_metadata(expert_id)
        if self._ensure_room_fn is not None:
            self._ensure_room_fn(metadata.size_bytes)

        hbm_pool = self.tier_manager.get_pool(MemoryTier.HBM)
        hbm_pool.store(expert_id, tensor)

        metadata.current_tier = MemoryTier.HBM
        metadata.transfer_state = TransferState.COMPLETE

        handle.request.state = TransferState.COMPLETE
        handle.request.completed_at = time.time()

        elapsed_ms = (time.perf_counter() - handle.start_time) * 1000
        self.metrics.record_transfer(handle.request.size_bytes, elapsed_ms)

        del self._inflight[expert_id]
        self._completed.add(expert_id)

        logger.debug(f"wait_for {expert_id}: completed in {elapsed_ms:.2f}ms")
        return tensor

    def poll_completed(self) -> List[ExpertId]:
        """Return expert IDs whose async transfers have finished."""
        done: List[ExpertId] = []
        for expert_id, handle in list(self._inflight.items()):
            is_ready = True
            if handle.event is not None and HAS_TORCH and torch.cuda.is_available():
                is_ready = handle.event.query()

            if is_ready:
                self.wait_for(expert_id)
                done.append(expert_id)
        return done

    def is_inflight(self, expert_id: ExpertId) -> bool:
        return expert_id in self._inflight

    def num_inflight(self) -> int:
        return len(self._inflight)

    def can_accept_transfer(self) -> bool:
        return self.num_inflight() < self.config.max_inflight_transfers

    def cancel(self, expert_id: ExpertId) -> bool:
        if expert_id in self._inflight:
            handle = self._inflight.pop(expert_id)
            handle.request.state = TransferState.FAILED
            self._staged_tensors.pop(expert_id, None)

            metadata = self.tier_manager.get_metadata(expert_id)
            metadata.transfer_state = TransferState.IDLE
            logger.debug(f"Cancelled transfer for {expert_id}")
            return True
        return False

    @property
    def double_buffer(self) -> DoubleBuffer:
        """Expose double-buffer for external overlap coordination."""
        return self._double_buffer

    # ── Internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _hop_count(source: MemoryTier, dest: MemoryTier) -> int:
        """How many tier-hops between *source* and *dest*."""
        from memtier_moe.memory.tier_manager import _TIER_ORDER
        return abs(_TIER_ORDER.index(dest) - _TIER_ORDER.index(source))

    def _create_transfer_request(self, expert_id: ExpertId) -> TransferRequest:
        metadata = self.tier_manager.get_metadata(expert_id)
        return TransferRequest(
            expert_id=expert_id,
            source_tier=metadata.current_tier,
            dest_tier=MemoryTier.HBM,
            size_bytes=metadata.size_bytes,
            priority=0,
            state=TransferState.IN_FLIGHT,
        )

    def _execute_multihop(
        self,
        expert_id: ExpertId,
        request: TransferRequest,
        hop_timings: List[HopTiming],
    ) -> Any:
        """Walk the expert upward through each tier until it reaches HBM."""
        metadata = self.tier_manager.get_metadata(expert_id)

        while metadata.current_tier != MemoryTier.HBM:
            hop_start = time.perf_counter()
            src = metadata.current_tier

            src_pool = self.tier_manager.get_pool(src)
            tensor = src_pool.retrieve(expert_id)
            src_pool.evict(expert_id)

            # Move to GPU if going to HBM, otherwise stay on CPU
            from memtier_moe.memory.tier_manager import _next_higher_tier
            dst = _next_higher_tier(src)
            dst_pool = self.tier_manager.get_pool(dst)

            if dst == MemoryTier.HBM and HAS_TORCH and isinstance(tensor, torch.Tensor) and torch.cuda.is_available():
                tensor = tensor.cuda()

            dst_pool.store(expert_id, tensor)
            metadata.current_tier = dst

            hop_ms = (time.perf_counter() - hop_start) * 1000
            hop_timings.append(HopTiming(source_tier=src, dest_tier=dst, duration_ms=hop_ms))

        self.metrics.record_promotion()
        return self.tier_manager.get_pool(MemoryTier.HBM).retrieve(expert_id)

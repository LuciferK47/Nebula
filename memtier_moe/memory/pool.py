"""Memory pool implementations for different hardware tiers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

from memtier_moe.core.types import ExpertId, MemoryTier

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


def _tensor_size_bytes(tensor: Any) -> int:
    """Calculate tensor or module size in bytes using duck typing."""
    if HAS_TORCH and isinstance(tensor, torch.nn.Module):
        param_bytes = sum(p.numel() * p.element_size() for p in tensor.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in tensor.buffers())
        return param_bytes + buffer_bytes
    if hasattr(tensor, 'parameters') and callable(getattr(tensor, 'parameters')):
        try:
            return sum(p.numel() * p.element_size() for p in tensor.parameters())
        except Exception:
            pass
    if hasattr(tensor, 'element_size') and hasattr(tensor, 'numel'):
        return tensor.element_size() * tensor.numel()
    return 1024  # fallback for opaque objects


def _move_to_device(tensor: Any, device: str, non_blocking: bool = False) -> Any:
    """Move tensor, module, or placeholder to target device."""
    if not HAS_TORCH:
        return tensor
    if isinstance(tensor, torch.nn.Module):
        if device == "cuda":
            if not torch.cuda.is_available():
                return tensor
            if non_blocking:
                for p in tensor.parameters():
                    p.data = p.data.to(device="cuda", non_blocking=True)
                for b in tensor.buffers():
                    b.data = b.data.to(device="cuda", non_blocking=True)
                return tensor
            return tensor.cuda()
        elif device == "cpu":
            return tensor.cpu()
    elif isinstance(tensor, torch.Tensor):
        if device == "cuda":
            return tensor.cuda(non_blocking=non_blocking) if torch.cuda.is_available() else tensor
        elif device == "cpu":
            return tensor.cpu()
    elif hasattr(tensor, device):
        try:
            return getattr(tensor, device)(non_blocking=non_blocking)
        except TypeError:
            return getattr(tensor, device)()
    return tensor

class MemoryPool(ABC):
    """Abstract base class for a memory pool managing tensors in a specific tier."""
    
    def __init__(self, capacity_bytes: int, tier: MemoryTier):
        self.capacity_bytes = capacity_bytes
        self.tier = tier

    @abstractmethod
    def store(self, expert_id: ExpertId, tensor: Any) -> None:
        """Store a tensor in this pool."""
        pass

    @abstractmethod
    def retrieve(self, expert_id: ExpertId) -> Any:
        """Retrieve a tensor from this pool."""
        pass

    @abstractmethod
    def evict(self, expert_id: ExpertId) -> Optional[Any]:
        """Evict a tensor from this pool, returning it."""
        pass

    @abstractmethod
    def contains(self, expert_id: ExpertId) -> bool:
        """Check if an expert is stored in this pool."""
        pass

    @abstractmethod
    def usage_bytes(self) -> int:
        """Return the current memory usage in bytes."""
        pass

    @abstractmethod
    def free_bytes(self) -> int:
        """Return the available free memory in bytes."""
        pass

    @abstractmethod
    def available_for(self, size_bytes: int) -> bool:
        """Check if the pool has enough free memory for the given size."""
        pass
        
    @property
    @abstractmethod
    def experts(self) -> List[ExpertId]:
        """Return a list of ExpertIds stored in this pool."""
        pass


class HBMPool(MemoryPool):
    """High Bandwidth Memory (GPU) pool implementation."""
    
    def __init__(self, capacity_bytes: int):
        super().__init__(capacity_bytes, MemoryTier.HBM)
        self._store: Dict[ExpertId, Any] = {}
        self._used_bytes: int = 0

    def store(self, expert_id: ExpertId, tensor: Any) -> None:
        tensor_size = _tensor_size_bytes(tensor)
        if not self.available_for(tensor_size):
            raise RuntimeError(f"HBM pool out of memory for expert {expert_id}")
            
        tensor = _move_to_device(tensor, "cuda")
                
        self._store[expert_id] = tensor
        self._used_bytes += tensor_size
        logger.debug(f"Stored expert {expert_id} in HBM. Usage: {self._used_bytes}/{self.capacity_bytes}")

    def retrieve(self, expert_id: ExpertId) -> Any:
        if expert_id not in self._store:
            raise KeyError(f"Expert {expert_id} not found in HBM pool")
        return self._store[expert_id]

    def evict(self, expert_id: ExpertId) -> Optional[Any]:
        if expert_id not in self._store:
            return None
        tensor = self._store.pop(expert_id)
        tensor_size = _tensor_size_bytes(tensor)
        self._used_bytes -= tensor_size
        logger.debug(f"Evicted expert {expert_id} from HBM. Usage: {self._used_bytes}/{self.capacity_bytes}")
        return tensor

    def contains(self, expert_id: ExpertId) -> bool:
        return expert_id in self._store

    def usage_bytes(self) -> int:
        return self._used_bytes

    def free_bytes(self) -> int:
        return self.capacity_bytes - self._used_bytes

    def available_for(self, size_bytes: int) -> bool:
        return self.free_bytes() >= size_bytes

    @property
    def experts(self) -> List[ExpertId]:
        return list(self._store.keys())


class DRAMPool(MemoryPool):
    """Host DRAM pool implementation with pinned memory support.

    Retrieval injects the tier's own access latency and applies a
    bandwidth-limited transfer cost via a token bucket, mirroring
    :class:`CXLPool`. Without this, DRAM hops were effectively free
    (zero-cost dict lookups), which is not representative of a real
    host-DRAM-to-HBM transfer over PCIe and made every offloading
    baseline look artificially fast.

    This charge is skipped (``retrieve(..., charge=False)``) when the
    caller is about to perform a *real* DRAM->HBM copy on real hardware
    (the live TieredMoEWrapper path) — the real CUDA transfer already
    measures that same PCIe hop, so charging both would double-count it.
    Pure-simulation callers (synthetic ``_Placeholder`` tensors, no real
    hardware transfer ever happens) keep the default ``charge=True``,
    since the injected cost is their only representation of that hop.
    """

    def __init__(
        self,
        capacity_bytes: int,
        latency_ns: int = 100,
        bandwidth_gbps: float = 16.0,
        burst_factor: float = 0.005,
    ):
        super().__init__(capacity_bytes, MemoryTier.DRAM)
        self._store: Dict[ExpertId, Any] = {}
        self._used_bytes: int = 0
        self.latency_ns = latency_ns
        self.bandwidth_gbps = bandwidth_gbps

        from memtier_moe.core.latency import TokenBucketRateLimiter
        self._rate_limiter = TokenBucketRateLimiter(bandwidth_gbps, burst_factor=burst_factor)
        # Match CXLPool: start empty so transfers are rate-limited to
        # bandwidth from the first request. The default burst_factor=1.5
        # on TokenBucketRateLimiter starts with ~1.5s of bandwidth already
        # banked (~24 GB at 16 GB/s) — effectively unlimited for any
        # realistic run — while CXLPool started empty, making DRAM look
        # artificially free relative to CXL for no physical reason.
        self._rate_limiter.tokens = 0.0

    def store(self, expert_id: ExpertId, tensor: Any) -> None:
        tensor_size = _tensor_size_bytes(tensor)
        if not self.available_for(tensor_size):
            raise RuntimeError(f"DRAM pool out of memory for expert {expert_id}")

        tensor = _move_to_device(tensor, "cpu")
        if HAS_TORCH and isinstance(tensor, torch.Tensor) and torch.cuda.is_available():
            try:
                tensor = tensor.pin_memory()
            except Exception:
                pass

        self._store[expert_id] = tensor
        self._used_bytes += tensor_size
        logger.debug(f"Stored expert {expert_id} in DRAM. Usage: {self._used_bytes}/{self.capacity_bytes}")

    def retrieve(self, expert_id: ExpertId, charge: bool = True) -> Any:
        """Retrieve a tensor, optionally injecting the modeled DRAM->HBM cost.

        Args:
            charge: When True (default), inject the synthetic latency and
                token-bucket bandwidth cost documented on this class. Pure
                simulation callers (no real tensor ever moves) need this —
                it is their only source of transfer cost. The live-model
                transfer engine passes ``charge=False`` for a real
                torch.Tensor/nn.Module about to make a real
                ``.to('cuda')`` hop: charging both here and paying the
                real PCIe copy would double-count the same physical DMA,
                inflating weight-transfer's measured cost by roughly 2x
                relative to what the hardware actually did.
        """
        if expert_id not in self._store:
            raise KeyError(f"Expert {expert_id} not found in DRAM pool")

        tensor = self._store[expert_id]
        if charge:
            from memtier_moe.core.latency import inject_latency_ns
            inject_latency_ns(self.latency_ns)
            self._rate_limiter.acquire(_tensor_size_bytes(tensor))
        return tensor

    def evict(self, expert_id: ExpertId) -> Optional[Any]:
        if expert_id not in self._store:
            return None
        tensor = self._store.pop(expert_id)
        tensor_size = _tensor_size_bytes(tensor)
        self._used_bytes -= tensor_size
        logger.debug(f"Evicted expert {expert_id} from DRAM. Usage: {self._used_bytes}/{self.capacity_bytes}")
        return tensor

    def contains(self, expert_id: ExpertId) -> bool:
        return expert_id in self._store

    def usage_bytes(self) -> int:
        return self._used_bytes

    def free_bytes(self) -> int:
        return self.capacity_bytes - self._used_bytes

    def available_for(self, size_bytes: int) -> bool:
        return self.free_bytes() >= size_bytes

    @property
    def experts(self) -> List[ExpertId]:
        return list(self._store.keys())


class CXLPool(MemoryPool):
    """CXL-attached memory pool with calibrated latency and bandwidth simulation.

    Stores tensors on CPU (like DRAM) but injects busy-wait latency and
    token-bucket bandwidth limiting on retrieve() to emulate CXL access
    characteristics.  The injected delays are approximate (~1-3× target)
    but preserve the relative ordering HBM < DRAM < CXL which is what
    matters for the tiering evaluation.
    """

    def __init__(
        self,
        capacity_bytes: int,
        latency_ns: int = 350,
        bandwidth_gbps: float = 8.0,
        burst_factor: float = 0.005,
        emulation_mode: str = "full",
    ) -> None:
        super().__init__(capacity_bytes, MemoryTier.CXL)
        self._store: Dict[ExpertId, Any] = {}
        self._used_bytes: int = 0
        self.latency_ns = latency_ns
        self.bandwidth_gbps = bandwidth_gbps

        # Import here to avoid circular — latency module is lightweight
        from memtier_moe.core.latency import TokenBucketRateLimiter
        self._rate_limiter = TokenBucketRateLimiter(bandwidth_gbps, burst_factor=burst_factor)
        self._rate_limiter.tokens = 0.0  # Start empty so transfers are rate-limited to bandwidth
        self.emulation_mode: str = emulation_mode  # "full", "latency_only", or "disabled"

    def store(self, expert_id: ExpertId, tensor: Any) -> None:
        tensor_size = _tensor_size_bytes(tensor)
        if not self.available_for(tensor_size):
            raise RuntimeError(f"CXL pool out of memory for expert {expert_id}")

        tensor = _move_to_device(tensor, "cpu")

        self._store[expert_id] = tensor
        self._used_bytes += tensor_size
        logger.debug(
            f"Stored expert {expert_id} in CXL. "
            f"Usage: {self._used_bytes}/{self.capacity_bytes}"
        )

    def emulate_access(self, size_bytes: int) -> float:
        """Inject calibrated CXL access latency and bus bandwidth delay based on emulation_mode."""
        mode = getattr(self, "emulation_mode", "full")
        if mode == "disabled":
            return 0.0

        from memtier_moe.core.latency import inject_latency_ns
        wait_ns = inject_latency_ns(self.latency_ns)

        if mode == "latency_only":
            return wait_ns / 1e9

        return self._rate_limiter.acquire(size_bytes)

    def retrieve(self, expert_id: ExpertId) -> Any:
        if expert_id not in self._store:
            raise KeyError(f"Expert {expert_id} not found in CXL pool")

        tensor = self._store[expert_id]
        tensor_size = _tensor_size_bytes(tensor)
        self.emulate_access(tensor_size)
        return tensor

    def evict(self, expert_id: ExpertId) -> Optional[Any]:
        if expert_id not in self._store:
            return None
        tensor = self._store.pop(expert_id)
        tensor_size = _tensor_size_bytes(tensor)
        self._used_bytes -= tensor_size
        logger.debug(
            f"Evicted expert {expert_id} from CXL. "
            f"Usage: {self._used_bytes}/{self.capacity_bytes}"
        )
        return tensor

    def contains(self, expert_id: ExpertId) -> bool:
        return expert_id in self._store

    def usage_bytes(self) -> int:
        return self._used_bytes

    def free_bytes(self) -> int:
        return self.capacity_bytes - self._used_bytes

    def available_for(self, size_bytes: int) -> bool:
        return self.free_bytes() >= size_bytes

    @property
    def experts(self) -> List[ExpertId]:
        return list(self._store.keys())

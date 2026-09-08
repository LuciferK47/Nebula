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
    """Calculate tensor size in bytes using duck typing."""
    if hasattr(tensor, 'element_size') and hasattr(tensor, 'numel'):
        return tensor.element_size() * tensor.numel()
    return 1024  # fallback for opaque objects

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
            
        if HAS_TORCH and isinstance(tensor, torch.Tensor):
            if torch.cuda.is_available():
                tensor = tensor.cuda()
                
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
    """

    def __init__(
        self,
        capacity_bytes: int,
        latency_ns: int = 100,
        bandwidth_gbps: float = 16.0,
    ):
        super().__init__(capacity_bytes, MemoryTier.DRAM)
        self._store: Dict[ExpertId, Any] = {}
        self._used_bytes: int = 0
        self.latency_ns = latency_ns
        self.bandwidth_gbps = bandwidth_gbps

        from memtier_moe.core.latency import TokenBucketRateLimiter
        self._rate_limiter = TokenBucketRateLimiter(bandwidth_gbps)

    def store(self, expert_id: ExpertId, tensor: Any) -> None:
        tensor_size = _tensor_size_bytes(tensor)
        if not self.available_for(tensor_size):
            raise RuntimeError(f"DRAM pool out of memory for expert {expert_id}")

        if HAS_TORCH and isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu()
            if torch.cuda.is_available():
                tensor = tensor.pin_memory()

        self._store[expert_id] = tensor
        self._used_bytes += tensor_size
        logger.debug(f"Stored expert {expert_id} in DRAM. Usage: {self._used_bytes}/{self.capacity_bytes}")

    def retrieve(self, expert_id: ExpertId) -> Any:
        if expert_id not in self._store:
            raise KeyError(f"Expert {expert_id} not found in DRAM pool")

        from memtier_moe.core.latency import inject_latency_ns
        inject_latency_ns(self.latency_ns)

        tensor = self._store[expert_id]
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
    ) -> None:
        super().__init__(capacity_bytes, MemoryTier.CXL)
        self._store: Dict[ExpertId, Any] = {}
        self._used_bytes: int = 0
        self.latency_ns = latency_ns
        self.bandwidth_gbps = bandwidth_gbps

        # Import here to avoid circular — latency module is lightweight
        from memtier_moe.core.latency import TokenBucketRateLimiter
        self._rate_limiter = TokenBucketRateLimiter(bandwidth_gbps)

    def store(self, expert_id: ExpertId, tensor: Any) -> None:
        tensor_size = _tensor_size_bytes(tensor)
        if not self.available_for(tensor_size):
            raise RuntimeError(f"CXL pool out of memory for expert {expert_id}")

        if HAS_TORCH and isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu()

        self._store[expert_id] = tensor
        self._used_bytes += tensor_size
        logger.debug(
            f"Stored expert {expert_id} in CXL. "
            f"Usage: {self._used_bytes}/{self.capacity_bytes}"
        )

    def retrieve(self, expert_id: ExpertId) -> Any:
        if expert_id not in self._store:
            raise KeyError(f"Expert {expert_id} not found in CXL pool")

        # Inject CXL access latency (busy-wait, NOT time.sleep)
        from memtier_moe.core.latency import inject_latency_ns
        inject_latency_ns(self.latency_ns)

        tensor = self._store[expert_id]

        # Simulate bandwidth constraint
        tensor_size = _tensor_size_bytes(tensor)
        self._rate_limiter.acquire(tensor_size)

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

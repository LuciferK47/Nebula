"""Unit tests for memory trace export to external simulators (DRAMSim3 / gem5)."""
import os
import tempfile
import pytest
import pandas as pd
from memtier_moe.core.types import MemoryTier
from memtier_moe.memory.trace_exporter import TraceExporter, MemoryTransaction


def test_trace_exporter_disabled_by_default():
    exporter = TraceExporter(enabled=False)
    exporter.record(tier=MemoryTier.HBM, size_bytes=4096, access_type="READ")
    assert exporter.transaction_count == 0


def test_trace_exporter_rejects_opcodes_the_export_formats_cant_parse():
    """DRAMSim3/gem5 trace formats only define READ/WRITE. A caller passing
    anything else (e.g. the old "TRANSFER" tag tiered_model.py used for
    hybrid-mode activation offload) used to be written through verbatim,
    silently producing an unparseable file — this caught 23.6% of one real
    exported trace. Case-normalized "read"/"write" must still pass."""
    exporter = TraceExporter(enabled=True)
    with pytest.raises(ValueError, match="READ or WRITE"):
        exporter.record(tier=MemoryTier.DRAM, size_bytes=4096, access_type="TRANSFER")
    assert exporter.transaction_count == 0

    exporter.record(tier=MemoryTier.DRAM, size_bytes=4096, access_type="read")
    exporter.record(tier=MemoryTier.DRAM, size_bytes=4096, access_type="write")
    assert exporter.transaction_count == 2


def test_trace_exporter_record():
    exporter = TraceExporter(enabled=True)
    exporter.record(
        tier=MemoryTier.HBM,
        size_bytes=17_301_504,
        access_type="WRITE",
        address=0x1000_0000,
        expert_id=(2, 3),
        tag="demand_fetch_weight",
    )
    exporter.record(
        tier=MemoryTier.DRAM,
        size_bytes=4096,
        access_type="READ",
        expert_id=(2, 5),
        tag="activation_offload",
    )
    assert exporter.transaction_count == 2


def test_trace_exporter_dramsim3_format():
    exporter = TraceExporter(enabled=True)
    exporter.record(tier=MemoryTier.HBM, size_bytes=64, access_type="READ", address=0x1000_2000)
    exporter.record(tier=MemoryTier.DRAM, size_bytes=64, access_type="WRITE", address=0x2000_4000)

    with tempfile.NamedTemporaryFile(suffix=".dramsim3", delete=False) as tmp:
        path = tmp.name

    try:
        count = exporter.export_dramsim3(path, clock_rate_ghz=2.0)
        assert count == 2
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2
        assert lines[0].startswith("0x000010002000 READ")
        assert lines[1].startswith("0x000020004000 WRITE")
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_trace_exporter_gem5_format():
    exporter = TraceExporter(enabled=True)
    exporter.record(tier=MemoryTier.CXL, size_bytes=256, access_type="READ", address=0x3000_1000)

    with tempfile.NamedTemporaryFile(suffix=".gem5", delete=False) as tmp:
        path = tmp.name

    try:
        count = exporter.export_gem5(path)
        assert count == 1
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
        parts = line.split()
        assert len(parts) == 5
        assert parts[1] == "READ"
        assert parts[2] == "0x30001000"
        assert parts[3] == "256"
        assert parts[4] == "0"
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_trace_exporter_pandas_and_csv():
    exporter = TraceExporter(enabled=True)
    exporter.record(tier=MemoryTier.HBM, size_bytes=4096, access_type="READ", expert_id=(0, 1), tag="hit")
    exporter.record(tier=MemoryTier.DRAM, size_bytes=4096, access_type="WRITE", expert_id=(0, 2), tag="miss")

    df = exporter.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "tier" in df.columns
    assert "size_bytes" in df.columns
    assert df["tier"].tolist() == ["HBM", "DRAM"]

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        path = tmp.name

    try:
        count = exporter.export_csv(path)
        assert count == 2
        df_csv = pd.read_csv(path)
        assert len(df_csv) == 2
        assert df_csv["tag"].tolist() == ["hit", "miss"]
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_trace_exporter_capacity_limit():
    exporter = TraceExporter(max_records=5, enabled=True)
    for i in range(10):
        exporter.record(tier=MemoryTier.HBM, size_bytes=64)
    assert exporter.transaction_count == 5


def test_trace_exporter_enable_api():
    """Test enable() method with optional capacity parameter."""
    exporter = TraceExporter(enabled=False, max_records=50)
    assert exporter.enabled is False

    exporter.enable(capacity=1000)
    assert exporter.enabled is True
    assert exporter.max_records == 1000

    exporter.record(tier=MemoryTier.HBM, size_bytes=128)
    assert exporter.transaction_count == 1

    exporter.stop_tracing()
    assert exporter.enabled is False


def test_trace_exporter_synthetic_addressing():
    """Verify synthetic deterministic memory mapping across memory tiers."""
    exporter = TraceExporter(enabled=True)
    # Tier base addresses:
    # HBM: 0x1000_0000_0000
    # DRAM: 0x2000_0000_0000
    # CXL: 0x3000_0000_0000
    # Expert offset: (layer * 64 + expert) * 32MB
    exporter.record(tier=MemoryTier.HBM, size_bytes=1024, expert_id=(0, 0))
    exporter.record(tier=MemoryTier.DRAM, size_bytes=1024, expert_id=(1, 2))
    exporter.record(tier=MemoryTier.CXL, size_bytes=1024, expert_id=(2, 0))

    tx0 = exporter.records[0]
    assert tx0.address == 0x1000_0000_0000

    tx1 = exporter.records[1]
    expected_dram_addr = 0x2000_0000_0000 + (1 * 64 + 2) * 32 * 1024 * 1024
    assert tx1.address == expected_dram_addr

    tx2 = exporter.records[2]
    expected_cxl_addr = 0x3000_0000_0000 + (2 * 64 + 0) * 32 * 1024 * 1024
    assert tx2.address == expected_cxl_addr


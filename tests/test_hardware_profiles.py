"""Unit tests for formal hardware specification profiles."""
import pytest
from memtier_moe.core.hardware_profiles import (
    HardwareProfile,
    PROFILES,
    get_hardware_profile,
    list_hardware_profiles,
    JEDEC_HBM3_CXL2,
    JEDEC_HBM3E_CXL3,
    WORKSTATION_RTX4050,
    DATACENTER_A100,
)
from memtier_moe.core.config import MemTierConfig


def test_list_hardware_profiles():
    profiles = list_hardware_profiles()
    assert "jedec-hbm3-cxl2" in profiles
    assert "jedec-hbm3e-cxl3" in profiles
    assert "workstation-rtx4050" in profiles
    assert "datacenter-a100" in profiles
    assert len(profiles) >= 4


def test_get_hardware_profile_valid():
    p1 = get_hardware_profile("JEDEC-HBM3-CXL2")
    assert p1.name == "jedec-hbm3-cxl2"
    assert p1.hbm_bandwidth_gbps == 819.2
    assert p1.cxl_latency_ns == 230

    p2 = get_hardware_profile("workstation-rtx4050")
    assert p2.gpu_vram_bytes == 6 * 1024 * 1024 * 1024
    assert p2.pcie_bandwidth_gbps == 16.0


def test_get_hardware_profile_invalid():
    with pytest.raises(KeyError, match="Unknown hardware profile"):
        get_hardware_profile("non_existent_profile_xyz")


def test_profile_create_config():
    p = JEDEC_HBM3E_CXL3
    cfg = p.create_config(hbm_cache_budget_bytes=64 * 1024 * 1024 * 1024)
    assert isinstance(cfg, MemTierConfig)
    assert cfg.gpu_vram_bytes == 96 * 1024 * 1024 * 1024
    assert cfg.hbm_cache_budget_bytes == 64 * 1024 * 1024 * 1024
    assert cfg.cxl_bandwidth_gbps == 64.0
    assert cfg.cxl_latency_ns == 180


def test_memtier_config_from_profile():
    cfg = MemTierConfig.from_profile("datacenter-a100", eviction_policy="size_aware")
    assert cfg.hardware_profile == "datacenter-a100"
    assert cfg.gpu_vram_bytes == 80 * 1024 * 1024 * 1024
    assert cfg.cxl_latency_ns == 260
    assert cfg.eviction_policy == "size_aware"

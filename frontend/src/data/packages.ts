import type { PackageCard } from '../types/content';

// The real package (pyproject.toml: name = "memtier-moe") is not published
// to PyPI — there is no "pip install memtier-moe" today. The previous
// copy invented "strata-moe" and two sibling packages that don't exist.
// This is the actual setup from Nebula/Readme.md.

export const installCommand = 'git clone https://github.com/LuciferK47/Nebula.git && cd Nebula && uv pip install -e .';

export const packages: PackageCard[] = [
{
  name: 'memtier_moe',
  command: 'uv pip install -e .',
  blurb: 'The tier manager, LFU cache, transfer engine and TieredMoEWrapper — everything under memtier_moe/. Editable install from source; not on PyPI.'
},
{
  name: 'scripts/run_scenarios.py',
  command: 'python scripts/run_scenarios.py --scenario all',
  blurb: 'The scenario suite behind every chart on this page — capacity sweeps, mode ablation, CXL sensitivity, and a 10-case edge-case probe.'
},
{
  name: 'scripts/serve.py',
  command: 'python scripts/serve.py',
  blurb: 'The API this page talks to when it can reach a GPU: /api/run, /api/compare, /api/historical_results.'
}];

export const usageSnippet = `from memtier_moe.core.config import MemTierConfig
from memtier_moe.runtime.tiered_model import TieredMoEWrapper

config = MemTierConfig(
    hbm_cache_budget_bytes=600 * 1024**2,   # the "expert cache budget"
    host_dram_bytes=1500 * 1024**2,
    cxl_memory_bytes=1500 * 1024**2,
)

wrapper = TieredMoEWrapper(
    model=base_model,               # any HF MoE model
    config=config,
    execution_mode="hybrid",        # activation offload, not weight swap
)

output_ids = wrapper.generate(**inputs, max_new_tokens=25)`;

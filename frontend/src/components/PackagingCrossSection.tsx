import React, { useState } from 'react';

interface LayerInfo {
  id: string;
  name: string;
  tech: string;
  bandwidth: string;
  latency: string;
  role: string;
  color: string;
}

const LAYERS: LayerInfo[] = [
  {
    id: 'compute',
    name: 'GPU Compute Die',
    tech: '4nm Custom Silicon',
    bandwidth: 'On-chip SRAM / Regs',
    latency: '< 1 ns',
    role: 'Hosts Streaming Multiprocessors, Tensor GEMM cores, and hardware MoE router.',
    color: '#e4512b',
  },
  {
    id: 'hbm',
    name: '4× HBM3e Memory Cubes',
    tech: '3D Stacked DRAM + TSVs',
    bandwidth: '3,200 GB/s (1024-bit per stack)',
    latency: '~10 ns',
    role: 'Primary fast tier. Holds hot expert weights pinned in GPU memory.',
    color: '#e4512b',
  },
  {
    id: 'interposer',
    name: '2.5D Silicon Interposer',
    tech: 'Passive High-Density Cu Wiring',
    bandwidth: 'Wide Parallel Micro-Bumps',
    latency: 'Sub-nanosecond',
    role: 'Ultra-dense interconnect bonding GPU compute die and HBM stacks with micro-bumps.',
    color: '#e5b52f',
  },
  {
    id: 'pcie',
    name: 'PCIe Gen5 x16 Interconnect',
    tech: 'PCIe 5.0 Root Complex',
    bandwidth: '64 GB/s Bi-directional',
    latency: '~100 ns PHY',
    role: 'Transfers weights and activations between GPU package and host DDR5 DRAM.',
    color: '#e5b52f',
  },
  {
    id: 'cxl',
    name: 'CXL 3.0 Far Memory Fabric',
    tech: 'CXL.mem Protocol over PCIe Physical',
    bandwidth: '8–32 GB/s Emulated',
    latency: '~350 ns Emulated',
    role: 'Coherent pooled memory tier allowing multi-host shared expert offload without OS paging.',
    color: '#8fa3b8',
  },
];

/**
 * Visual packaging cross-section diagram inspired by ByteByteGo & ArchitectureNotes.
 * Allows interactive inspection of the physical hierarchy from silicon die to CXL fabric.
 */
export function PackagingCrossSection() {
  const [selectedLayer, setSelectedLayer] = useState<string>('compute');
  const active = LAYERS.find((l) => l.id === selectedLayer) || LAYERS[0];

  return (
    <div className="mt-14 rounded-2xl border border-ink-line bg-ink-panel p-6 md:p-8">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-ink-line pb-4">
        <div>
          <span className="font-mono text-10 font-bold uppercase tracking-wider text-amber">
            Physical Silicon & Packaging Cross-Section
          </span>
          <h4 className="mt-1 font-display text-xl text-cream md:text-2xl">
            From Silicon Die to Disaggregated Fabric
          </h4>
        </div>
        <div className="font-mono text-[10px] text-khaki/60">
          Click any layer to inspect physical characteristics
        </div>
      </div>

      <div className="mt-6 grid gap-8 lg:grid-cols-[1fr_20rem]">
        {/* Interactive Vertical Packaging Stack Diagram */}
        <div className="space-y-2.5">
          {LAYERS.map((layer) => {
            const isSelected = selectedLayer === layer.id;
            return (
              <button
                key={layer.id}
                type="button"
                onClick={() => setSelectedLayer(layer.id)}
                className={`w-full rounded-xl border p-3.5 text-left transition-all duration-200 ${
                  isSelected
                    ? 'border-cream bg-ink-soft shadow-lg'
                    : 'border-ink-line bg-ink/60 hover:border-cream/40 hover:bg-ink'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span
                      className="h-3 w-3 rounded-full"
                      style={{ backgroundColor: layer.color }}
                    />
                    <span className="font-mono text-xs font-bold text-cream">
                      {layer.name}
                    </span>
                    <span className="rounded bg-white/5 px-2 py-0.5 font-mono text-[9px] text-khaki/70">
                      {layer.tech}
                    </span>
                  </div>
                  <div className="font-mono text-[10px] font-semibold" style={{ color: layer.color }}>
                    {layer.bandwidth}
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Detail Inspection Card */}
        <div className="rounded-xl border border-ink-line bg-ink p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <span
                className="font-mono text-[10px] font-bold uppercase tracking-wider"
                style={{ color: active.color }}
              >
                {active.tech}
              </span>
              <span className="font-mono text-[10px] text-khaki/50">Spec Sheet</span>
            </div>

            <h5 className="mt-2 font-display text-2xl text-cream">{active.name}</h5>

            <p className="mt-3 font-mono text-xs leading-relaxed text-khaki/80">
              {active.role}
            </p>

            <div className="mt-6 space-y-2 border-t border-ink-line pt-4 font-mono text-xs">
              <div className="flex justify-between text-cream/70">
                <span className="text-khaki/60">Bandwidth:</span>
                <span className="font-semibold text-cream">{active.bandwidth}</span>
              </div>
              <div className="flex justify-between text-cream/70">
                <span className="text-khaki/60">Target Latency:</span>
                <span className="font-semibold text-cream">{active.latency}</span>
              </div>
            </div>
          </div>

          <div className="mt-6 rounded-lg bg-white/5 p-3 font-mono text-[10px] text-khaki/70">
            <strong className="text-amber">Key Insight: </strong>
            Moving a 16.5 MB expert across PCIe/CXL incurs significant latency. Nebula minimizes this by combining LFU frequency caching with dynamic activation offloading.
          </div>
        </div>
      </div>
    </div>
  );
}

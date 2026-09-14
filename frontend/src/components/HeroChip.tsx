import React from 'react';

const TIER_META = [
  {
    id: 'hbm',
    label: 'GPU HBM3e',
    sub: '3.2 TB/s · Co-Packaged Silicon',
    y: 20,
    cells: 12,
    lit: 5,
    dotColor: '#e4512b',
    barColor: '#e4512b',
  },
  {
    id: 'dram',
    label: 'Host DDR5 DRAM',
    sub: '64 GB/s · PCIe Gen5',
    y: 112,
    cells: 12,
    lit: 3,
    dotColor: '#d97706',
    barColor: '#e5b52f',
  },
  {
    id: 'cxl',
    label: 'CXL 3.0 Far Memory',
    sub: '8–32 GB/s · Coherent Fabric',
    y: 204,
    cells: 12,
    lit: 4,
    dotColor: '#3b6998',
    barColor: '#8fa3b8',
  },
] as const;

/**
 * Editorial chip & multi-tier memory hierarchy diagram.
 * Visualizes:
 * - 2.5D Advanced Silicon Packaging: GPU Compute Die & 4x HBM3e memory stacks on Silicon Interposer
 * - High-speed interconnect conduits (PCIe Gen5 x16 & CXL 3.0 Fabric)
 * - Three tiered memory residency layers with live animated data flow pulses
 */
export function HeroChip() {
  const skewY = -10;
  const plateStartX = 295;

  return (
    <figure className="relative mx-auto w-full max-w-[54rem]">
      {/* Container with background color matching the page seamlessly, no grey/checkered patch */}
      <div className="relative rounded-2xl border border-ink/10 bg-cream p-4 md:p-8 shadow-[0_2px_16px_-2px_rgba(20,20,20,0.06)]">
        <svg
          viewBox="0 0 860 310"
          className="w-full select-none"
          role="img"
          aria-label="Physical packaging and multi-tier memory residency hierarchy diagram"
        >
          <defs>
            {/* Linear gradients for packaging layers */}
            <linearGradient id="die-grad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#25272e" />
              <stop offset="100%" stopColor="#121316" />
            </linearGradient>
            <linearGradient id="hbm-grad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#e4512b" stopOpacity="0.9" />
              <stop offset="100%" stopColor="#962d14" stopOpacity="0.9" />
            </linearGradient>
            <linearGradient id="interposer-grad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#1c1e24" />
              <stop offset="50%" stopColor="#2e313b" />
              <stop offset="100%" stopColor="#1c1e24" />
            </linearGradient>

            {/* Pulse Keyframe Animation */}
            <style>{`
              @keyframes pulseArc {
                0% { stroke-dashoffset: 260; opacity: 0.2; }
                50% { opacity: 0.95; }
                100% { stroke-dashoffset: 0; opacity: 0.2; }
              }
              @keyframes pulseDrop {
                0% { transform: translateY(0); opacity: 0.2; }
                50% { opacity: 1; }
                100% { transform: translateY(85px); opacity: 0.2; }
              }
              .pulse-weight {
                stroke-dasharray: 45, 200;
                animation: pulseArc 2.4s cubic-bezier(0.4, 0, 0.2, 1) infinite;
              }
            `}</style>
          </defs>

          {/* Clean Left Column: Distinct, High-Contrast Labels & Specs */}
          {TIER_META.map((tier) => (
            <g key={`label-${tier.id}`} transform={`translate(16, ${tier.y + 12})`}>
              {/* Vibrant Indicator Dot */}
              <circle cx={14} cy={14} r={5.5} fill={tier.dotColor} />
              
              {/* Primary Tier Name - Solid Ink on Cream for maximum legibility */}
              <text
                x={30}
                y={13}
                fontFamily="IBM Plex Mono, monospace"
                fontSize={13.5}
                fontWeight="700"
                fill="#141414"
                letterSpacing="0.03em"
              >
                {tier.label}
              </text>

              {/* Bandwidth & Interconnect Spec - High-contrast charcoal with clear typography */}
              <text
                x={30}
                y={30}
                fontFamily="IBM Plex Mono, monospace"
                fontSize={11}
                fontWeight="600"
                fill="#374151"
                letterSpacing="0.01em"
              >
                {tier.sub}
              </text>
            </g>
          ))}

          {/* PCIe / CXL Interconnect Bus Backplane Pillars */}
          <g transform={`translate(${plateStartX}, 35)`}>
            <line x1={20} y1={20} x2={20} y2={215} stroke="#d1d5db" strokeWidth={5} strokeLinecap="round" />
            <line x1={20} y1={20} x2={20} y2={215} stroke="#d97706" strokeWidth={2} strokeDasharray="3 4" opacity={0.8} />
            <line x1={450} y1={20} x2={450} y2={215} stroke="#d1d5db" strokeWidth={5} strokeLinecap="round" />
            <line x1={450} y1={20} x2={450} y2={215} stroke="#3b6998" strokeWidth={2} strokeDasharray="3 4" opacity={0.8} />
          </g>

          {/* Three Stacked Physical Memory Tiers */}
          <g transform={`translate(${plateStartX}, 30)`}>
            {TIER_META.map((tier) => (
              <g key={tier.id} transform={`translate(0, ${tier.y}) skewX(${skewY})`}>
                {/* Hardware Carrier Base Plate */}
                <rect
                  x={-8}
                  y={-4}
                  width={486}
                  height={64}
                  rx={8}
                  fill="#18191e"
                  stroke="#2b2e38"
                  strokeWidth={1.5}
                />
                <rect
                  x={0}
                  y={0}
                  width={470}
                  height={56}
                  rx={6}
                  fill={tier.id === 'hbm' ? 'url(#interposer-grad)' : '#20222a'}
                  opacity={0.96}
                />

                {/* Micro copper trace detailing on HBM Interposer */}
                {tier.id === 'hbm' && (
                  <g opacity={0.3}>
                    <line x1={20} y1={12} x2={450} y2={12} stroke="#e5b52f" strokeWidth={0.9} strokeDasharray="4 6" />
                    <line x1={20} y1={44} x2={450} y2={44} stroke="#e5b52f" strokeWidth={0.9} strokeDasharray="4 6" />
                  </g>
                )}

                {/* Expert Memory Cells */}
                {Array.from({ length: tier.cells }).map((_, i) => {
                  const lit = i < tier.lit;
                  return (
                    <g key={i}>
                      <rect
                        x={16 + i * 37}
                        y={10}
                        width={29}
                        height={36}
                        rx={3.5}
                        fill={lit ? tier.barColor : '#2b2d35'}
                        stroke={lit ? '#ffffff44' : '#3d404c'}
                        strokeWidth={0.8}
                        opacity={lit ? 1 : 0.65}
                      />
                      {lit && (
                        <circle cx={16 + i * 37 + 14.5} cy={16} r={1.6} fill="#ffffff" opacity={0.9} />
                      )}
                    </g>
                  );
                })}
              </g>
            ))}
          </g>

          {/* Weight promotion arc: CXL -> HBM (16.5MB expert promotion) */}
          <path
            d={`M ${plateStartX + 110} 242 C ${plateStartX + 180} 180, ${plateStartX + 180} 100, ${plateStartX + 110} 55`}
            fill="none"
            stroke="#64748b"
            strokeWidth={2}
            strokeDasharray="4 4"
            opacity={0.4}
          />
          <path
            d={`M ${plateStartX + 110} 242 C ${plateStartX + 180} 180, ${plateStartX + 180} 100, ${plateStartX + 110} 55`}
            fill="none"
            stroke="#2563eb"
            strokeWidth={2.8}
            className="pulse-weight"
          />
          <circle cx={plateStartX + 110} cy={55} r={4.5} fill="#2563eb" />
          <circle cx={plateStartX + 110} cy={242} r={4} fill="#64748b" />

          {/* Activation offload drop line (Hybrid mode: 4-8KB activation transfer) */}
          <path
            d={`M ${plateStartX + 300} 60 L ${plateStartX + 300} 150`}
            fill="none"
            stroke="#7c3aed"
            strokeWidth={2.2}
            strokeDasharray="3 3"
            opacity={0.65}
          />
          <circle cx={plateStartX + 300} cy={150} r={4} fill="#7c3aed" />

          {/* High-Contrast Data Flow Annotation Badges */}
          <g transform={`translate(${plateStartX + 165}, 125)`}>
            <rect
              x={-8}
              y={-11}
              width={138}
              height={22}
              rx={5}
              fill="#18191e"
              stroke="#3b82f6"
              strokeWidth={1}
            />
            <text
              x={61}
              y={4}
              textAnchor="middle"
              fontFamily="IBM Plex Mono, monospace"
              fontSize={9.5}
              fill="#93c5fd"
              fontWeight="600"
              letterSpacing="0.04em"
            >
              WEIGHT: ~16.5 MB
            </text>
          </g>

          <g transform={`translate(${plateStartX + 315}, 100)`}>
            <rect
              x={-6}
              y={-11}
              width={142}
              height={22}
              rx={5}
              fill="#18191e"
              stroke="#8b5cf6"
              strokeWidth={1}
            />
            <text
              x={65}
              y={4}
              textAnchor="middle"
              fontFamily="IBM Plex Mono, monospace"
              fontSize={9.5}
              fill="#c4b5fd"
              fontWeight="600"
              letterSpacing="0.04em"
            >
              ACTIVATION: ~4–8 KB
            </text>
          </g>
        </svg>
      </div>

      <figcaption className="mt-3.5 flex flex-wrap items-center justify-between gap-2 font-mono text-10 uppercase tracking-label text-ink/60">
        <span className="font-semibold">fig. 01 — physical packaging & memory residency</span>
        <span className="flex items-center gap-4">
          <span className="flex items-center gap-1.5 font-medium">
            <span className="h-2 w-2 rounded-full bg-hot" /> HBM
          </span>
          <span className="flex items-center gap-1.5 font-medium">
            <span className="h-2 w-2 rounded-full bg-amber" /> DRAM
          </span>
          <span className="flex items-center gap-1.5 font-medium">
            <span className="h-2 w-2 rounded-full bg-[#3b6998]" /> CXL
          </span>
        </span>
      </figcaption>
    </figure>
  );
}

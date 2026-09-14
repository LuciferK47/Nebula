import React, { useState, useEffect } from 'react';
import { NUM_LAYERS, EXPERTS_PER_LAYER, TIER_COLOR, TOTAL_EXPERTS, type TierId } from '../../three/layout';

const TIER_ORDER: TierId[] = ['hbm', 'dram', 'cxl'];
const TIER_NAME: Record<TierId, string> = { hbm: 'HBM', dram: 'DRAM', cxl: 'CXL' };

/** Static SVG fallback for reduced-motion, small screens, or a lost WebGL
 * context — same real placement data as the 3D scene, laid out as three
 * flat rows instead of a rotatable stack. Not a "sorry, unsupported"
 * apology: this carries the same information. */
export function ChipDiagram2D({
  placement,
  focusTier,
  mode,
  playToken,
}: {
  placement: TierId[];
  focusTier?: TierId | null;
  mode?: 'weight_transfer' | 'hybrid';
  playToken?: number;
}) {
  const [lastAction, setLastAction] = useState<string | null>(null);

  useEffect(() => {
    if (!playToken || playToken === 0) return;
    const msg = mode === 'weight_transfer'
      ? '⚡ Weight Promotion: 16.5 MB migrated to HBM (evicting resident block to DRAM)'
      : '⚡ Hybrid Offload: 4 KB activation dispatched to resident expert in DRAM';
    setLastAction(msg);
    const timer = setTimeout(() => setLastAction(null), 3000);
    return () => clearTimeout(timer);
  }, [playToken, mode]);

  const rowsByTier: Record<TierId, number> = { hbm: 0, dram: 0, cxl: 0 };
  placement.forEach((t) => rowsByTier[t]++);

  const cell = 14;
  const gap = 2;
  const cols = EXPERTS_PER_LAYER * NUM_LAYERS;
  const rowW = cols * (cell + gap);

  return (
    <div className="w-full rounded-lg bg-ink p-6 transition-all duration-200">
      {lastAction && (
        <div className="mb-4 animate-pulse rounded border border-amber/50 bg-amber/10 px-3 py-1.5 font-mono text-[11px] text-amber">
          {lastAction}
        </div>
      )}

      <svg
        viewBox={`0 0 ${rowW + 40} 200`}
        className="w-full"
        role="img"
        aria-label="Expert placement across HBM, DRAM and CXL tiers"
      >
        {TIER_ORDER.map((tier, rowIdx) => {
          const y = 20 + rowIdx * 60;
          let x = 20;
          const isDimmed = focusTier != null && focusTier !== tier;
          const isFocused = focusTier === tier;

          return (
            <g
              key={tier}
              className="transition-opacity duration-200"
              style={{ opacity: isDimmed ? 0.25 : 1 }}
            >
              <text
                x={0}
                y={y + 10}
                fontSize={10}
                fill={TIER_COLOR[tier]}
                fontWeight={isFocused ? 'bold' : 'normal'}
                className="font-mono"
              >
                {TIER_NAME[tier]}
                {isFocused ? ' ★' : ''}
              </text>
              {placement.map((t, i) => {
                if (t !== tier) return null;
                const rect = (
                  <rect
                    key={i}
                    x={x}
                    y={y}
                    width={cell}
                    height={cell}
                    rx={2}
                    fill={TIER_COLOR[tier]}
                    stroke={isFocused ? '#ffffff' : 'none'}
                    strokeWidth={isFocused ? 1 : 0}
                  />
                );
                x += cell + gap;
                return rect;
              })}
            </g>
          );
        })}
      </svg>
      <div className="mt-4 flex flex-wrap items-center justify-between gap-2 font-mono text-10 uppercase tracking-label text-khaki/60">
        <span>
          {TOTAL_EXPERTS} experts across {NUM_LAYERS} layers — {rowsByTier.hbm} HBM /{' '}
          {rowsByTier.dram} DRAM / {rowsByTier.cxl} CXL
        </span>
        {focusTier && (
          <span className="text-amber">
            Focus: {TIER_NAME[focusTier]} ({rowsByTier[focusTier]} resident)
          </span>
        )}
      </div>
    </div>
  );
}

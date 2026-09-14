import { NUM_LAYERS, EXPERTS_PER_LAYER, TIER_COLOR, TOTAL_EXPERTS, type TierId } from '../../three/layout';

const TIER_ORDER: TierId[] = ['hbm', 'dram', 'cxl'];
const TIER_NAME: Record<TierId, string> = { hbm: 'HBM', dram: 'DRAM', cxl: 'CXL' };

/** Static SVG fallback for reduced-motion, small screens, or a lost WebGL
 * context — same real placement data as the 3D scene, laid out as three
 * flat rows instead of a rotatable stack. Not a "sorry, unsupported"
 * apology: this carries the same information. */
export function ChipDiagram2D({ placement }: { placement: TierId[] }) {
  const rowsByTier: Record<TierId, number> = { hbm: 0, dram: 0, cxl: 0 };
  placement.forEach((t) => rowsByTier[t]++);

  const cell = 14;
  const gap = 2;
  const cols = EXPERTS_PER_LAYER * NUM_LAYERS;
  const rowW = cols * (cell + gap);

  return (
    <div className="rounded-lg bg-ink p-6">
      <svg viewBox={`0 0 ${rowW + 40} 200`} className="w-full" role="img" aria-label="Expert placement across HBM, DRAM and CXL tiers">
        {TIER_ORDER.map((tier, rowIdx) => {
          const y = 20 + rowIdx * 60;
          let x = 20;
          return (
            <g key={tier}>
              <text x={0} y={y + 10} fontSize={10} fill={TIER_COLOR[tier]} className="font-mono">
                {TIER_NAME[tier]}
              </text>
              {placement.map((t, i) => {
                if (t !== tier) return null;
                const rect = <rect key={i} x={x} y={y} width={cell} height={cell} rx={2} fill={TIER_COLOR[tier]} />;
                x += cell + gap;
                return rect;
              })}
            </g>);

        })}
      </svg>
      <p className="mt-4 font-mono text-10 uppercase tracking-label text-khaki/50">
        {TOTAL_EXPERTS} experts across {NUM_LAYERS} layers — {rowsByTier.hbm} HBM /{' '}
        {rowsByTier.dram} DRAM / {rowsByTier.cxl} CXL
      </p>
    </div>);

}

import { useState } from 'react';
import { Html } from '@react-three/drei';
import { PLATE_DEPTH, PLATE_WIDTH, TIER_COLOR, TIER_Y, type TierId } from './layout';

const TIER_SPECS: Record<
  TierId,
  { name: string; subtitle: string; bw: string; lat: string; type: string }
> = {
  hbm: {
    name: 'GPU VRAM',
    subtitle: 'Primary Accelerator Residency',
    bw: '> 192 GB/s',
    lat: '< 10 ns',
    type: 'High-speed local GPU memory for resident hot experts',
  },
  dram: {
    name: 'Host DRAM',
    subtitle: 'Warm System Memory Tier',
    bw: '32–64 GB/s',
    lat: '~100 ns',
    type: 'Pinned system memory for warm staging and CPU compute',
  },
  cxl: {
    name: 'CXL Far Memory',
    subtitle: 'Disaggregated Coherent Pool',
    bw: '8–32 GB/s',
    lat: '~350 ns',
    type: 'Pooled far memory expander over coherent CXL fabric',
  },
};

export function TierPlates({
  explode,
  focusTier,
}: {
  explode: number;
  focusTier: TierId | null;
}) {
  const tiers: TierId[] = ['hbm', 'dram', 'cxl'];
  const [hoveredTier, setHoveredTier] = useState<TierId | null>(null);

  return (
    <>
      {tiers.map((tier) => {
        const y = TIER_Y[tier] * (1 + explode * 0.9);
        const dimmed = focusTier != null && focusTier !== tier;
        const spec = TIER_SPECS[tier];
        const isHovered = hoveredTier === tier;

        return (
          <group
            key={tier}
            position={[0, y, 0]}
            onPointerOver={(e) => {
              e.stopPropagation();
              setHoveredTier(tier);
            }}
            onPointerOut={() => setHoveredTier(null)}
          >
            {/* Main Hardware Tier Chassis Plate (Realistic Dark PCB Substrate) */}
            <mesh position={[0, -0.07, 0]}>
              <boxGeometry args={[PLATE_WIDTH, 0.12, PLATE_DEPTH]} />
              <meshStandardMaterial
                color={dimmed ? '#0f1012' : '#16181d'}
                roughness={0.4}
                metalness={0.7}
                transparent
                opacity={dimmed ? 0.3 : 0.98}
              />
            </mesh>

            {/* Beveled Tier Accent Rim */}
            <mesh position={[0, -0.005, 0]}>
              <boxGeometry args={[PLATE_WIDTH * 0.99, 0.015, PLATE_DEPTH * 0.99]} />
              <meshStandardMaterial
                color={TIER_COLOR[tier]}
                emissive={TIER_COLOR[tier]}
                emissiveIntensity={dimmed ? 0.05 : 0.25}
                roughness={0.3}
                transparent
                opacity={dimmed ? 0.25 : 0.75}
              />
            </mesh>

            {/* Realistic DRAM Surface: Dual rows of Black BGA Memory IC Packages with Gold Edge Pins */}
            {tier === 'dram' && (
              <group position={[0, 0.015, 0]}>
                {/* 8 DDR5 BGA Memory Packages */}
                {Array.from({ length: 8 }).map((_, idx) => {
                  const row = idx < 4 ? -0.9 : 0.9;
                  const col = (idx % 4) * 1.8 - 2.7;
                  return (
                    <group key={`dram-ic-${idx}`} position={[col, 0, row]}>
                      <mesh position={[0, 0.02, 0]}>
                        <boxGeometry args={[1.2, 0.04, 0.8]} />
                        <meshStandardMaterial color="#1a1c23" roughness={0.3} metalness={0.5} />
                      </mesh>
                      {/* Silver Pin-1 Notch Dot */}
                      <mesh position={[-0.45, 0.045, -0.25]}>
                        <cylinderGeometry args={[0.04, 0.04, 0.01, 8]} />
                        <meshStandardMaterial color="#cbd5e1" metalness={0.9} roughness={0.2} />
                      </mesh>
                    </group>
                  );
                })}
                {/* Gold PCIe / DIMM Edge Connector Fingers */}
                <mesh position={[0, -0.02, PLATE_DEPTH / 2 - 0.08]}>
                  <boxGeometry args={[PLATE_WIDTH * 0.88, 0.02, 0.14]} />
                  <meshStandardMaterial color="#e5b52f" metalness={0.85} roughness={0.2} />
                </mesh>
              </group>
            )}

            {/* Realistic CXL Surface: Central CXL 3.0 Controller ASIC Heatsink + Far Memory Buffers */}
            {tier === 'cxl' && (
              <group position={[0, 0.02, 0]}>
                {/* Central CXL 3.0 Protocol Controller with Heat Sink Fins */}
                <mesh position={[0, 0.06, 0]}>
                  <boxGeometry args={[2.0, 0.1, 1.8]} />
                  <meshStandardMaterial color="#2d3748" roughness={0.25} metalness={0.8} />
                </mesh>
                {Array.from({ length: 6 }).map((_, f) => (
                  <mesh key={`cxl-fin-${f}`} position={[(f - 2.5) * 0.32, 0.12, 0]}>
                    <boxGeometry args={[0.06, 0.05, 1.7]} />
                    <meshStandardMaterial color="#4a5568" roughness={0.2} metalness={0.85} />
                  </mesh>
                ))}
                {/* Far Memory Buffer Chips */}
                {[-2.8, 2.8].map((xOffset) => (
                  <mesh key={`buf-${xOffset}`} position={[xOffset, 0.025, 0]}>
                    <boxGeometry args={[1.4, 0.05, 1.4]} />
                    <meshStandardMaterial color="#1e293b" roughness={0.35} metalness={0.6} />
                  </mesh>
                ))}
                {/* High-speed PCIe Gen5 Edge Connector */}
                <mesh position={[0, -0.02, PLATE_DEPTH / 2 - 0.08]}>
                  <boxGeometry args={[PLATE_WIDTH * 0.9, 0.02, 0.14]} />
                  <meshStandardMaterial color="#e5b52f" metalness={0.85} roughness={0.2} />
                </mesh>
              </group>
            )}

            {/* Realistic GPU VRAM Surface: High-speed Memory Interface Traces */}
            {tier === 'hbm' && (
              <group position={[0, 0.015, 0]}>
                {/* Corner High-Density Memory Stacks */}
                {[
                  [-2.8, -1.2],
                  [2.8, -1.2],
                  [-2.8, 1.2],
                  [2.8, 1.2],
                ].map(([cx, cz], i) => (
                  <group key={`hbm-stack-${i}`} position={[cx, 0.05, cz]}>
                    <mesh position={[0, 0, 0]}>
                      <boxGeometry args={[1.2, 0.1, 1.2]} />
                      <meshStandardMaterial color="#b91c1c" roughness={0.3} metalness={0.6} />
                    </mesh>
                  </group>
                ))}
              </group>
            )}

            {/* Detailed Hover Inspection Card (Visible ONLY on hover) */}
            {isHovered && !dimmed && (
              <Html
                position={[0, 0.75, 0]}
                center
                style={{ pointerEvents: 'none' }}
              >
                <div
                  className="whitespace-nowrap rounded-lg bg-[#0c0d10]/95 px-3 py-2 font-mono text-xs text-cream shadow-2xl border transition-all"
                  style={{ borderColor: `${TIER_COLOR[tier]}88`, boxShadow: `0 0 20px ${TIER_COLOR[tier]}33` }}
                >
                  <div className="flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full" style={{ backgroundColor: TIER_COLOR[tier] }} />
                    <span className="font-bold text-xs" style={{ color: TIER_COLOR[tier] }}>{spec.name}</span>
                    <span className="text-white/40">·</span>
                    <span className="text-cream/80 text-[11px]">{spec.subtitle}</span>
                  </div>
                  <div className="text-cream/70 text-[10px] mt-1">{spec.type}</div>
                  <div className="mt-1.5 flex gap-4 text-[10px] text-cream/60 border-t border-white/10 pt-1">
                    <span>
                      Bandwidth: <strong className="text-cream">{spec.bw}</strong>
                    </span>
                    <span>
                      Latency: <strong className="text-cream">{spec.lat}</strong>
                    </span>
                  </div>
                </div>
              </Html>
            )}
          </group>
        );
      })}
    </>
  );
}

import React, { useState } from 'react';
import { Html } from '@react-three/drei';
import { PLATE_DEPTH, PLATE_WIDTH, TIER_COLOR, TIER_Y, type TierId } from './layout';

const TIER_SPECS: Record<
  TierId,
  { name: string; subtitle: string; bw: string; lat: string; type: string }
> = {
  hbm: {
    name: 'GPU HBM3e',
    subtitle: 'On-Package Resident Pool',
    bw: '819–3200 GB/s',
    lat: '~10 ns',
    type: 'Co-packaged with GPU Die via 2.5D Silicon Interposer',
  },
  dram: {
    name: 'Host DDR5 DRAM',
    subtitle: 'PCIe Gen5 System Memory',
    bw: '16–64 GB/s',
    lat: '~100 ns',
    type: 'Dual-Channel DDR5 DIMMs via CPU PCIe Gen5 Root Complex',
  },
  cxl: {
    name: 'CXL 3.0 Pooled Memory',
    subtitle: 'Coherent Far Memory Tier',
    bw: '8–32 GB/s',
    lat: '~350 ns (emulated)',
    type: 'CXL 3.0 Type-3 Memory Expander over PCIe Fabric',
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
            {/* Main Hardware Tier Chassis Plate */}
            <mesh position={[0, -0.07, 0]}>
              <boxGeometry args={[PLATE_WIDTH, 0.14, PLATE_DEPTH]} />
              <meshStandardMaterial
                color={dimmed ? '#111214' : '#1a1b1f'}
                roughness={0.5}
                metalness={0.6}
                transparent
                opacity={dimmed ? 0.35 : 0.96}
              />
            </mesh>

            {/* Beveled Tier Accent Border Rim */}
            <mesh position={[0, -0.005, 0]}>
              <boxGeometry args={[PLATE_WIDTH * 0.99, 0.02, PLATE_DEPTH * 0.99]} />
              <meshStandardMaterial
                color={TIER_COLOR[tier]}
                emissive={TIER_COLOR[tier]}
                emissiveIntensity={dimmed ? 0.05 : 0.22}
                roughness={0.3}
                transparent
                opacity={dimmed ? 0.3 : 0.8}
              />
            </mesh>

            {/* Tier-Specific Hardware Details */}
            {tier === 'dram' && (
              /* DRAM DIMM Notch lines and gold contact fingers on the edge */
              <group position={[0, 0.01, PLATE_DEPTH / 2 - 0.15]}>
                <mesh position={[0, 0, 0]}>
                  <boxGeometry args={[PLATE_WIDTH * 0.85, 0.02, 0.12]} />
                  <meshStandardMaterial
                    color="#e5b52f"
                    metalness={0.8}
                    roughness={0.2}
                  />
                </mesh>
              </group>
            )}

            {tier === 'cxl' && (
              /* CXL AIC Heatsink Fins / PCIe Bracket representation */
              <group position={[0, 0.04, -PLATE_DEPTH / 2 + 0.3]}>
                {Array.from({ length: 9 }).map((_, fin) => (
                  <mesh key={`fin-${fin}`} position={[(fin - 4) * 0.9, 0, 0]}>
                    <boxGeometry args={[0.08, 0.08, 0.5]} />
                    <meshStandardMaterial
                      color="#3a3e47"
                      roughness={0.3}
                      metalness={0.7}
                    />
                  </mesh>
                ))}
              </group>
            )}

            {/* Left Hardware Specification Label Badge */}
            <Html
              position={[-PLATE_WIDTH / 2 - 1.2, 0.3, 0]}
              center
              style={{ pointerEvents: 'none' }}
            >
              <div
                className="whitespace-nowrap rounded-lg px-3 py-1.5 font-mono shadow-xl transition-all duration-200"
                style={{
                  backgroundColor: 'rgba(20, 20, 20, 0.92)',
                  border: `1px solid ${TIER_COLOR[tier]}${dimmed ? '33' : '88'}`,
                  opacity: dimmed ? 0.35 : 1,
                }}
              >
                <div className="flex items-center gap-2">
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{ backgroundColor: TIER_COLOR[tier] }}
                  />
                  <span
                    className="text-[11px] font-bold uppercase tracking-wider"
                    style={{ color: TIER_COLOR[tier] }}
                  >
                    {spec.name}
                  </span>
                </div>
                <div className="mt-0.5 text-[9px] text-khaki/70">
                  {spec.bw} · {spec.lat}
                </div>
              </div>
            </Html>

            {/* Hover Detailed Modal */}
            {isHovered && !dimmed && (
              <Html
                position={[0, 0.8, 0]}
                center
                style={{ pointerEvents: 'none' }}
              >
                <div className="whitespace-nowrap rounded-lg bg-ink/95 px-3 py-2 font-mono text-[10px] text-cream shadow-2xl border border-ink-line">
                  <div className="font-bold text-amber">{spec.subtitle}</div>
                  <div className="text-khaki/80">{spec.type}</div>
                  <div className="mt-1 flex gap-4 text-[9px] text-cream/60">
                    <span>Bandwidth: <strong className="text-cream">{spec.bw}</strong></span>
                    <span>Latency: <strong className="text-cream">{spec.lat}</strong></span>
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

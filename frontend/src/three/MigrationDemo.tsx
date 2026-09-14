import React, { useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { CELL_SIZE_X, CELL_SIZE_Z, DIE_Y, TIER_Y } from './layout';

type Mode = 'weight_transfer' | 'hybrid';

const easeInOutCubic = (t: number) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

/**
 * The contrast that is this project's actual thesis, animated directly
 * rather than only asserted in a chart: weight_transfer physically hauls
 * a 16.5 MB expert between tiers (and evicts something to make room);
 * hybrid mode only ever moves a few KB of activation down to where the
 * expert already sits, computes there, and sends the result back — the
 * expert itself never moves.
 */
export function MigrationDemo({ mode, playToken, explode }: { mode: Mode; playToken: number; explode: number }) {
  const [progress, setProgress] = useState(1);
  const startRef = useRef(0);

  React.useEffect(() => {
    if (playToken === 0) return;
    startRef.current = performance.now();
    setProgress(0);
  }, [playToken]);

  useFrame(() => {
    if (progress >= 1) return;
    const elapsed = (performance.now() - startRef.current) / 1000;
    const duration = mode === 'weight_transfer' ? 1.6 : 1.1;
    setProgress(Math.min(elapsed / duration, 1));
  });

  const weightGeom = useMemo(() => new THREE.BoxGeometry(CELL_SIZE_X, 0.32, CELL_SIZE_Z), []);
  const activationGeom = useMemo(() => new THREE.SphereGeometry(0.14, 16, 16), []);

  if (playToken === 0) return null;

  const t = easeInOutCubic(progress);
  const scale = 1 + explode * 0.9;

  if (mode === 'weight_transfer') {
    // The promoted expert: CXL -> HBM, an arc through DRAM height.
    const fromY = TIER_Y.cxl * scale;
    const toY = TIER_Y.hbm * scale;
    const midY = TIER_Y.dram * scale + 1.4;
    const x = THREE.MathUtils.lerp(-2.6, 2.6, t);
    const y = (1 - t) * (1 - t) * fromY + 2 * (1 - t) * t * midY + t * t * toY;

    // The evicted expert: HBM -> DRAM, the opposite direction, slightly offset.
    const evictFromY = TIER_Y.hbm * scale;
    const evictToY = TIER_Y.dram * scale;
    const ey = THREE.MathUtils.lerp(evictFromY, evictToY, t);

    return (
      <>
        <mesh position={[x, y + 0.5, 0.6]} geometry={weightGeom}>
          <meshStandardMaterial color="#8fa3b8" emissive="#8fa3b8" emissiveIntensity={0.3} />
        </mesh>
        <mesh position={[2.8, ey + 0.5, -0.6]} geometry={weightGeom}>
          <meshStandardMaterial color="#e4512b" emissive="#e4512b" emissiveIntensity={0.2} transparent opacity={0.8} />
        </mesh>
      </>);

  }

  // hybrid: an activation drops from the die to a DRAM cell and returns.
  // The round trip is symmetric: 0->0.5 down, 0.5->1 back up.
  const downT = t < 0.5 ? t * 2 : 1;
  const upT = t > 0.5 ? (t - 0.5) * 2 : 0;
  const y = t < 0.5 ?
  THREE.MathUtils.lerp(DIE_Y * scale, TIER_Y.dram * scale + 0.5, easeInOutCubic(downT)) :
  THREE.MathUtils.lerp(TIER_Y.dram * scale + 0.5, DIE_Y * scale, easeInOutCubic(upT));

  // Violet, not amber: this sphere lands on the yellow DRAM plate, and amber
  // (#f5b32b) is a near-fail against the DRAM tier yellow (#e5b52f) under
  // deutan/protan color vision (ΔE 2.5 - see HeroChip.tsx for the same fix).
  return (
    <mesh position={[0.8, y, 0.3]} geometry={activationGeom}>
      <meshStandardMaterial color="#7c6bd6" emissive="#7c6bd6" emissiveIntensity={0.6} />
    </mesh>);

}

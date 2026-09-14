import React, { useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import { DIE_Y, TIER_COLOR, TIER_Y } from './layout';

interface BusLinesProps {
  explode: number;
}

/**
 * Animated High-Speed Interconnect Fabric:
 * - PCIe Gen5 x16 Bus Links connecting GPU Package to Host DDR5 DRAM (64 GB/s)
 * - CXL 3.0 Fabric Links connecting Host to CXL Pooled Memory (8–32 GB/s, 350ns)
 * - Flowing data particle packets simulating bus transactions and memory coherency snoops
 */
export function BusLines({ explode }: BusLinesProps) {
  const scale = 1 + explode * 0.9;
  const pciePacketRef = useRef<THREE.Mesh>(null!);
  const cxlPacketRef = useRef<THREE.Mesh>(null!);
  const [hoveredBus, setHoveredBus] = useState<'pcie' | 'cxl' | null>(null);

  // Bus pillar coordinates (Left and Right pillars)
  const pillars = useMemo(
    () => [
      { id: 'pillar-l-front', x: -4.8, z: -3.8 },
      { id: 'pillar-r-front', x: 4.8, z: -3.8 },
      { id: 'pillar-l-back', x: -4.8, z: 3.8 },
      { id: 'pillar-r-back', x: 4.8, z: 3.8 },
    ],
    []
  );

  const topY = (DIE_Y - 0.7) * scale;
  const hbmY = TIER_Y.hbm * scale;
  const dramY = TIER_Y.dram * scale;
  const cxlY = TIER_Y.cxl * scale;

  // Animate bus packets
  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();

    // PCIe Packet ping-pongs between HBM and DRAM
    if (pciePacketRef.current) {
      const cycle = (t * 1.8) % 2;
      const progress = cycle < 1 ? cycle : 2 - cycle;
      pciePacketRef.current.position.y = THREE.MathUtils.lerp(hbmY, dramY, progress);
    }

    // CXL Packet ping-pongs between DRAM and CXL
    if (cxlPacketRef.current) {
      const cycle = (t * 1.2 + 0.5) % 2;
      const progress = cycle < 1 ? cycle : 2 - cycle;
      cxlPacketRef.current.position.y = THREE.MathUtils.lerp(dramY, cxlY, progress);
    }
  });

  return (
    <group>
      {/* Structural PCIe / CXL High-Speed Bus Pillars */}
      {pillars.map((p) => {
        const height = topY - cxlY;
        const midY = (topY + cxlY) / 2;
        return (
          <group key={p.id} position={[p.x, midY, p.z]}>
            {/* Bus conduit tube */}
            <mesh>
              <cylinderGeometry args={[0.07, 0.07, height, 16]} />
              <meshStandardMaterial
                color="#282a30"
                roughness={0.3}
                metalness={0.8}
              />
            </mesh>
            {/* Bus core fiber line */}
            <mesh>
              <cylinderGeometry args={[0.03, 0.03, height + 0.05, 12]} />
              <meshStandardMaterial
                color="#e5b52f"
                emissive="#e5b52f"
                emissiveIntensity={0.2}
              />
            </mesh>
          </group>
        );
      })}

      {/* Animated PCIe Packet (Amber) traversing between HBM and DRAM */}
      <mesh
        ref={pciePacketRef}
        position={[-4.8, (hbmY + dramY) / 2, -3.8]}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHoveredBus('pcie');
        }}
        onPointerOut={() => setHoveredBus(null)}
      >
        <sphereGeometry args={[0.18, 16, 16]} />
        <meshStandardMaterial
          color="#e5b52f"
          emissive="#e5b52f"
          emissiveIntensity={0.9}
        />
      </mesh>

      {/* Animated CXL Packet (Cold Cyan/Slate) traversing between DRAM and CXL */}
      <mesh
        ref={cxlPacketRef}
        position={[4.8, (dramY + cxlY) / 2, -3.8]}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHoveredBus('cxl');
        }}
        onPointerOut={() => setHoveredBus(null)}
      >
        <sphereGeometry args={[0.18, 16, 16]} />
        <meshStandardMaterial
          color="#8fa3b8"
          emissive="#8fa3b8"
          emissiveIntensity={0.9}
        />
      </mesh>

      {/* Bus Labels (Visible ONLY on hover) */}
      {hoveredBus === 'pcie' && (
        <Html
          position={[-5.6, (hbmY + dramY) / 2, -3.8]}
          center
          style={{ pointerEvents: 'none' }}
        >
          <div className="whitespace-nowrap rounded-md bg-ink/95 px-2.5 py-1 font-mono text-[9px] uppercase tracking-wider text-amber border border-amber/40 shadow-xl">
            PCIe Gen5 x16 (64 GB/s Interconnect)
          </div>
        </Html>
      )}

      {hoveredBus === 'cxl' && (
        <Html
          position={[5.6, (dramY + cxlY) / 2, -3.8]}
          center
          style={{ pointerEvents: 'none' }}
        >
          <div className="whitespace-nowrap rounded-md bg-ink/95 px-2.5 py-1 font-mono text-[9px] uppercase tracking-wider text-cold border border-cold/40 shadow-xl">
            CXL 3.0 Coherent Memory Fabric (8–32 GB/s)
          </div>
        </Html>
      )}
    </group>
  );
}

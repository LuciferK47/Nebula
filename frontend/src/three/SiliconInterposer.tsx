import React, { useMemo } from 'react';
import * as THREE from 'three';
import { Html } from '@react-three/drei';
import { DIE_Y, PLATE_DEPTH, PLATE_WIDTH } from './layout';

interface SiliconInterposerProps {
  explode: number;
}

/**
 * 2.5D Silicon Interposer Layer with micro-bumps and etched sub-micron interconnects.
 * Provides the ultra-wide physical memory bus routing between the GPU compute die
 * and flanking HBM3e stacks.
 */
export function SiliconInterposer({ explode }: SiliconInterposerProps) {
  // Sits directly below the Die and HBM stacks
  const interposerY = (DIE_Y - 0.28) * (1 + explode * 0.9);
  const interposerW = PLATE_WIDTH * 0.82;
  const interposerD = PLATE_DEPTH * 0.68;
  const thickness = 0.12;

  const [hovered, setHovered] = React.useState(false);

  // Micro-channel bus traces connecting central die to HBM cubes
  const traceLines = useMemo(() => {
    const lines: [number, number, number, number][] = [];
    // Left bus traces
    for (let i = -3; i <= 3; i++) {
      lines.push([-2.2, i * 0.3, -0.9, i * 0.3]);
      lines.push([0.9, i * 0.3, 2.2, i * 0.3]);
    }
    return lines;
  }, []);

  return (
    <group position={[0, interposerY, 0]}>
      {/* Silicon Interposer Base Plate */}
      <mesh
        position={[0, -thickness / 2, 0]}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
        }}
        onPointerOut={() => setHovered(false)}
      >
        <boxGeometry args={[interposerW, thickness, interposerD]} />
        <meshStandardMaterial
          color="#16181d"
          roughness={0.2}
          metalness={0.7}
          wireframeLinewidth={1}
        />
      </mesh>

      {/* Gold Micro-bump Array Interface (top face) */}
      <mesh position={[0, 0.005, 0]}>
        <boxGeometry args={[interposerW * 0.97, 0.01, interposerD * 0.97]} />
        <meshStandardMaterial
          color="#22242a"
          roughness={0.4}
          metalness={0.8}
        />
      </mesh>

      {/* Etched High-Density Interconnect Micro-traces (Gold/Amber) */}
      {traceLines.map(([x1, z1, x2, z2], idx) => (
        <mesh key={`trace-${idx}`} position={[(x1 + x2) / 2, 0.015, (z1 + z2) / 2]}>
          <boxGeometry args={[Math.abs(x2 - x1), 0.008, 0.04]} />
          <meshStandardMaterial
            color="#e5b52f"
            emissive="#e5b52f"
            emissiveIntensity={0.3}
            roughness={0.3}
            metalness={0.9}
          />
        </mesh>
      ))}

      {/* Sub-micron routing grid accent */}
      <mesh position={[0, 0.012, 0]}>
        <planeGeometry args={[interposerW * 0.9, interposerD * 0.9]} />
        <meshBasicMaterial color="#e5b52f" wireframe transparent opacity={0.08} />
      </mesh>

      {hovered && (
        <Html position={[0, 0.5, 0]} center style={{ pointerEvents: 'none' }}>
          <div className="whitespace-nowrap rounded-md bg-ink px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-wide text-cream shadow-xl border border-ink-line">
            <span className="text-amber">2.5D SILICON INTERPOSER · </span>
            <span>Micro-Bumps & Passive Cu Interconnects (3.2 TB/s)</span>
          </div>
        </Html>
      )}
    </group>
  );
}

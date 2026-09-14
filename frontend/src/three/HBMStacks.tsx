import React, { useState } from 'react';
import { Html } from '@react-three/drei';
import { DIE_Y, PLATE_DEPTH, PLATE_WIDTH, TIER_COLOR } from './layout';

interface HBMStacksProps {
  explode: number;
}

/**
 * 3D HBM3e Memory Stacks flanking the GPU Compute Die on the Silicon Interposer.
 * Models 4 high-bandwidth memory cubes:
 * - 4x vertically stacked DRAM dies per stack
 * - Vertical TSV (Through-Silicon Via) wiring representation
 * - Base Logic/Buffer die interfacing with the interposer
 * - Over 3.2 TB/s aggregated memory bandwidth
 */
export function HBMStacks({ explode }: HBMStacksProps) {
  const y = DIE_Y * (1 + explode * 0.9);
  const dieW = PLATE_WIDTH * 0.42;
  const stackW = 1.15;
  const stackD = 1.15;
  const stackH = 0.38;
  const numLayers = 4;
  const layerH = stackH / numLayers;

  const [hoveredStack, setHoveredStack] = useState<string | null>(null);

  // Positions for 4 HBM3e stacks: 2 on Left, 2 on Right of GPU Die
  const stackPositions: { id: string; label: string; x: number; z: number }[] = [
    { id: 'hbm-0', label: 'HBM3e Stack #0 (24GB)', x: -dieW / 2 - stackW / 2 - 0.28, z: -stackD / 2 - 0.2 },
    { id: 'hbm-1', label: 'HBM3e Stack #1 (24GB)', x: -dieW / 2 - stackW / 2 - 0.28, z: stackD / 2 + 0.2 },
    { id: 'hbm-2', label: 'HBM3e Stack #2 (24GB)', x: dieW / 2 + stackW / 2 + 0.28, z: -stackD / 2 - 0.2 },
    { id: 'hbm-3', label: 'HBM3e Stack #3 (24GB)', x: dieW / 2 + stackW / 2 + 0.28, z: stackD / 2 + 0.2 },
  ];

  return (
    <group position={[0, y, 0]}>
      {stackPositions.map((stack) => (
        <group
          key={stack.id}
          position={[stack.x, 0, stack.z]}
          onPointerOver={(e) => {
            e.stopPropagation();
            setHoveredStack(`${stack.label} · 819 GB/s Bandwidth (1024-bit Bus)`);
          }}
          onPointerOut={() => setHoveredStack(null)}
        >
          {/* Base Logic/Interface Die */}
          <mesh position={[0, 0.04, 0]}>
            <boxGeometry args={[stackW * 1.05, 0.08, stackD * 1.05]} />
            <meshStandardMaterial color="#2d2d2a" roughness={0.3} metalness={0.6} />
          </mesh>

          {/* Stacked DRAM Dies with TSV separator cuts */}
          {Array.from({ length: numLayers }).map((_, l) => {
            const layerY = 0.08 + l * layerH + layerH / 2;
            return (
              <mesh key={`die-${stack.id}-${l}`} position={[0, layerY, 0]}>
                <boxGeometry args={[stackW, layerH * 0.88, stackD]} />
                <meshStandardMaterial
                  color="#c85a32"
                  roughness={0.2}
                  metalness={0.8}
                />
              </mesh>
            );
          })}

          {/* Protective Top Cap */}
          <mesh position={[0, stackH + 0.06, 0]}>
            <boxGeometry args={[stackW * 0.98, 0.02, stackD * 0.98]} />
            <meshStandardMaterial color="#1a1a18" roughness={0.4} metalness={0.7} />
          </mesh>

          {/* Micro-bump Interconnect Pads beneath base die */}
          <mesh position={[0, 0.01, 0]}>
            <boxGeometry args={[stackW * 0.9, 0.02, stackD * 0.9]} />
            <meshStandardMaterial
              color="#e5b52f"
              metalness={0.9}
              roughness={0.1}
            />
          </mesh>
        </group>
      ))}

      {/* Hover Info Tooltip */}
      {hoveredStack && (
        <Html position={[0, stackH + 0.8, 0]} center style={{ pointerEvents: 'none' }}>
          <div className="whitespace-nowrap rounded-md bg-ink px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-wide text-cream shadow-xl border border-ink-line">
            <span style={{ color: TIER_COLOR.hbm }}>HBM3e · </span>
            <span>{hoveredStack}</span>
          </div>
        </Html>
      )}
    </group>
  );
}

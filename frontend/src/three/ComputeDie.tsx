import React from 'react';
import { Html } from '@react-three/drei';
import { DIE_Y, PLATE_DEPTH, PLATE_WIDTH } from './layout';

interface ComputeDieProps {
  explode: number;
}

/**
 * High-fidelity 3D GPU Compute Die inspired by ArchitectureNotes & ByteByteGo chip diagrams.
 * Visualizes the modern accelerator die:
 * - SM (Streaming Multiprocessor) / Tensor Core arrays
 * - Central L2 Cache partition
 * - Hardware MoE Router / Top-k Gate Dispatcher
 * - High-speed PCIe Gen5 / CXL PHY controller
 */
export function ComputeDie({ explode }: ComputeDieProps) {
  const y = DIE_Y * (1 + explode * 0.9);
  const dieW = PLATE_WIDTH * 0.42;
  const dieD = PLATE_DEPTH * 0.42;
  const dieH = 0.35;

  // Grid of SM (Streaming Multiprocessors) / Tensor Cores
  const smRows = 3;
  const smCols = 4;
  const smGap = 0.08;
  const coreW = (dieW * 0.85 - (smCols - 1) * smGap) / smCols;
  const coreD = (dieD * 0.35 - (smRows - 1) * smGap) / smRows;

  const [hoveredBlock, setHoveredBlock] = React.useState<string | null>(null);

  return (
    <group position={[0, y, 0]}>
      {/* Primary Silicon Substrate / Die Base */}
      <mesh position={[0, dieH / 2, 0]}>
        <boxGeometry args={[dieW, dieH, dieD]} />
        <meshStandardMaterial
          color="#1e1f24"
          roughness={0.25}
          metalness={0.7}
          envMapIntensity={0.8}
        />
      </mesh>

      {/* Die Surface Bevel / Protective Guard Ring */}
      <mesh position={[0, dieH + 0.01, 0]}>
        <boxGeometry args={[dieW * 0.96, 0.02, dieD * 0.96]} />
        <meshStandardMaterial color="#121316" roughness={0.4} metalness={0.5} />
      </mesh>

      {/* Top SM / Tensor Core Cluster (Left Bank) */}
      {Array.from({ length: smRows }).map((_, r) =>
        Array.from({ length: smCols }).map((_, c) => {
          const xPos = -dieW * 0.38 + c * (coreW + smGap) + coreW / 2;
          const zPos = -dieD * 0.26 + r * (coreD + smGap) + coreD / 2;
          return (
            <mesh
              key={`sm-top-${r}-${c}`}
              position={[xPos, dieH + 0.03, zPos]}
              onPointerOver={(e) => {
                e.stopPropagation();
                setHoveredBlock('Matrix Compute Units');
              }}
              onPointerOut={() => setHoveredBlock(null)}
            >
              <boxGeometry args={[coreW, 0.03, coreD]} />
              <meshStandardMaterial
                color="#2a2c33"
                roughness={0.3}
                metalness={0.6}
              />
            </mesh>
          );
        })
      )}

      {/* Bottom SM / Tensor Core Cluster (Right Bank) */}
      {Array.from({ length: smRows }).map((_, r) =>
        Array.from({ length: smCols }).map((_, c) => {
          const xPos = -dieW * 0.38 + c * (coreW + smGap) + coreW / 2;
          const zPos = dieD * 0.08 + r * (coreD + smGap) + coreD / 2;
          return (
            <mesh
              key={`sm-bot-${r}-${c}`}
              position={[xPos, dieH + 0.03, zPos]}
              onPointerOver={(e) => {
                e.stopPropagation();
                setHoveredBlock('Matrix Compute Units');
              }}
              onPointerOut={() => setHoveredBlock(null)}
            >
              <boxGeometry args={[coreW, 0.03, coreD]} />
              <meshStandardMaterial
                color="#2a2c33"
                roughness={0.3}
                metalness={0.6}
              />
            </mesh>
          );
        })
      )}

      {/* Central High-Speed Cache Strip */}
      <mesh
        position={[0, dieH + 0.03, -0.06]}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHoveredBlock('Fast Shared Cache');
        }}
        onPointerOut={() => setHoveredBlock(null)}
      >
        <boxGeometry args={[dieW * 0.88, 0.035, dieD * 0.12]} />
        <meshStandardMaterial
          color="#383c48"
          roughness={0.2}
          metalness={0.8}
        />
      </mesh>

      {/* MoE Gate / Router Dispatcher Block */}
      <mesh
        position={[-dieW * 0.18, dieH + 0.04, -0.06]}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHoveredBlock('MoE Dynamic Router & Gate');
        }}
        onPointerOut={() => setHoveredBlock(null)}
      >
        <boxGeometry args={[dieW * 0.26, 0.04, dieD * 0.1]} />
        <meshStandardMaterial
          color="#e4512b"
          emissive="#e4512b"
          emissiveIntensity={0.4}
          roughness={0.3}
          metalness={0.4}
        />
      </mesh>

      {/* High-speed CXL / PCIe Interface */}
      <mesh
        position={[0, dieH + 0.025, dieD * 0.44]}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHoveredBlock('CXL & PCIe High-Speed Interface');
        }}
        onPointerOut={() => setHoveredBlock(null)}
      >
        <boxGeometry args={[dieW * 0.8, 0.025, dieD * 0.06]} />
        <meshStandardMaterial
          color="#e5b52f"
          emissive="#e5b52f"
          emissiveIntensity={0.25}
          roughness={0.3}
        />
      </mesh>

      {/* Component Hover Tooltip */}
      {hoveredBlock && (
        <Html
          position={[0, dieH + 0.9, 0]}
          center
          style={{ pointerEvents: 'none' }}
        >
          <div className="whitespace-nowrap rounded-md bg-ink px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-wide text-cream shadow-xl border border-ink-line">
            <span className="text-amber">COMPUTE · </span>
            <span>{hoveredBlock}</span>
          </div>
        </Html>
      )}

      {/* Overall Layer Label */}
      <Html
        position={[0, dieH + 0.45, -dieD / 2 - 0.3]}
        center
        style={{ pointerEvents: 'none' }}
      >
        <div className="rounded-full bg-ink/90 px-3 py-1 font-mono text-[10px] font-bold uppercase tracking-wider text-cream border border-white/10 backdrop-blur-sm">
          Compute Layer
        </div>
      </Html>
    </group>
  );
}

import React, { useLayoutEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { Html } from '@react-three/drei';
import {
  CELL_HEIGHT,
  CELL_SIZE_X,
  CELL_SIZE_Z,
  EXPERTS_PER_LAYER,
  NUM_LAYERS,
  TIER_COLOR,
  TIER_Y,
  TOTAL_EXPERTS,
  cellIndex,
  cellX,
  cellZ,
  tmpColor,
  tmpMatrix,
  tmpPosition,
  tmpQuaternion,
  tmpScale,
  type TierId } from
'./layout';

interface HoverInfo {
  layer: number;
  expert: number;
  tier: TierId;
}

export function ExpertField({
  placement,
  explode


}: {placement: TierId[];explode: number;}) {
  const meshRef = useRef<THREE.InstancedMesh>(null!);
  const [hover, setHover] = React.useState<HoverInfo | null>(null);
  const geometry = useMemo(() => new THREE.BoxGeometry(1, 1, 1, 1, 1, 1), []);
  const material = useMemo(
    () => new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.08 }),
    []
  );

  useLayoutEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    for (let layer = 0; layer < NUM_LAYERS; layer++) {
      for (let expert = 0; expert < EXPERTS_PER_LAYER; expert++) {
        const i = cellIndex(layer, expert);
        const tier = placement[i];
        const y = TIER_Y[tier] * (1 + explode * 0.9) + CELL_HEIGHT / 2 + 0.2;
        tmpPosition.set(cellX(expert), y, cellZ(layer));
        tmpScale.set(CELL_SIZE_X, CELL_HEIGHT, CELL_SIZE_Z);
        tmpMatrix.compose(tmpPosition, tmpQuaternion, tmpScale);
        mesh.setMatrixAt(i, tmpMatrix);
        tmpColor.set(TIER_COLOR[tier]);
        mesh.setColorAt(i, tmpColor);
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [placement, explode]);

  return (
    <>
      <instancedMesh
        ref={meshRef}
        args={[geometry, material, TOTAL_EXPERTS]}
        onPointerMove={(e) => {
          e.stopPropagation();
          if (e.instanceId == null) return;
          const layer = Math.floor(e.instanceId / EXPERTS_PER_LAYER);
          const expert = e.instanceId % EXPERTS_PER_LAYER;
          setHover({ layer, expert, tier: placement[e.instanceId] });
        }}
        onPointerOut={() => setHover(null)} />

      {hover &&
      <Html
        position={[
        cellX(hover.expert),
        TIER_Y[hover.tier] * (1 + explode * 0.9) + CELL_HEIGHT + 0.6,
        cellZ(hover.layer)]
        }
        center
        style={{ pointerEvents: 'none' }}>

          <div className="rounded-md bg-ink px-3 py-2 font-mono text-[10px] uppercase tracking-wide text-cream shadow-lg">
            <div className="text-cream/50">
              layer {hover.layer} · expert {hover.expert}
            </div>
            <div style={{ color: TIER_COLOR[hover.tier] }}>{hover.tier.toUpperCase()}</div>
          </div>
        </Html>
      }
    </>);

}

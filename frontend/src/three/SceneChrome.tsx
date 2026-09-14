/** Key + fill + rim lighting with accent lighting for silicon die, HBM stacks and interconnects. */
export function SceneLights() {
  return (
    <>
      <hemisphereLight args={['#f4f3ed', '#141414', 0.65]} />
      <directionalLight position={[9, 14, 8]} intensity={1.2} />
      <directionalLight position={[-8, 6, -9]} intensity={0.45} color="#8fa3b8" />
      <pointLight position={[0, 7, 0]} intensity={0.4} color="#e5b52f" distance={12} />
    </>
  );
}

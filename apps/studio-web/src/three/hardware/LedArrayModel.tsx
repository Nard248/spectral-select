import { useMemo } from 'react'
import * as THREE from 'three'
import { RoundedBox } from '@react-three/drei'
import { Anodized, Alu, Chrome, Rubber, Plastic } from '../util/materials'
import { MountRig, Clamp, HullMat, hullColor } from '../util/parts'
import { labelPlate } from '../util/textures'
import { glowColor, mountFor, outwardXZ, type SourcePartProps } from './sourceCommon'

/** LED array: ring of LED dies on a white MCPCB disc, finned heat-sink behind, small driver box on the table. */
export function LedArrayModel({ source, wavelengthNm, q, selected, hovered }: SourcePartProps) {
  const R = Math.max(28, source.beam.aperture_diameter_mm / 2 + 8)
  const hull = hullColor(selected, hovered)
  const glow = glowColor(wavelengthNm, 1.9)
  const dim = new THREE.Color('#c9a22c').multiplyScalar(0.55)
  const mount = useMemo(() => mountFor(source, q, [0, R + 5, 26]), [source, q, R])
  const dies = useMemo(() => {
    const pts: [number, number][] = [[0, 0]]
    ;[[6, 0.38], [12, 0.74]].forEach(([n, rr]) => { for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2 + (n === 12 ? 0.26 : 0); pts.push([Math.cos(a) * R * rr, Math.sin(a) * R * rr]) } })
    return pts
  }, [R])
  const fins = useMemo(() => Array.from({ length: 20 }, (_, i) => (i / 20) * Math.PI), [])
  const driver = useMemo(() => {
    const o = outwardXZ(source)
    const p = source.position_mm
    return { x: o.x * 150, z: o.y * 150 + 0, ang: Math.atan2(o.x, o.y), y: -p[1] }
  }, [source])
  const cable = useMemo(() => {
    const gy = -source.position_mm[1]
    return new THREE.TubeGeometry(new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0, 55).applyQuaternion(q),
      new THREE.Vector3(0, 0, 90).applyQuaternion(q).add(new THREE.Vector3(0, -30, 0)),
      new THREE.Vector3(driver.x * 0.6, gy + 16, driver.z * 0.6),
      new THREE.Vector3(driver.x, gy + 20, driver.z),
    ]), 60, 2.6, 10, false)
  }, [q, driver, source.position_mm])
  const plate = labelPlate(['LED-12 Array', 'Multi-line 365 – 660 nm'], { w: 512, h: 128 })
  return (
    <group>
      <group quaternion={q}>
        {/* PCB + bezel */}
        <mesh position={[0, 0, 1.2]} rotation={[Math.PI / 2, 0, 0]} castShadow><cylinderGeometry args={[R, R, 2.4, 64]} />
          <meshPhysicalMaterial color="#eef1f5" roughness={0.35} metalness={0.05} clearcoat={0.5} /></mesh>
        <mesh position={[0, 0, 0.2]} rotation={[0, 0, 0]}><torusGeometry args={[R + 1, 2.2, 14, 64]} /><Chrome color="#cdd3dc" /></mesh>
        {dies.map(([x, y], i) => (
          <group key={i} position={[x, y, -0.2]}>
            <mesh position={[0, 0, 0.7]}><boxGeometry args={[6.2, 6.2, 1.4]} /><meshPhysicalMaterial color="#f7f4ea" roughness={0.3} /></mesh>
            <mesh position={[0, 0, -0.1]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, 1, 0.6]}><sphereGeometry args={[2.1, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2]} />
              <meshBasicMaterial color={wavelengthNm == null ? dim : glow} toneMapped={false} /></mesh>
          </group>
        ))}
        {/* heat-sink */}
        <mesh position={[0, 0, 5.5]} rotation={[Math.PI / 2, 0, 0]} castShadow><cylinderGeometry args={[R, R, 6, 64]} /><Anodized color="#101216" /></mesh>
        {fins.map((a, i) => (
          <mesh key={i} position={[0, 0, 8 + 22]} rotation={[0, 0, a]} castShadow>
            <boxGeometry args={[R * 2 - 4, 1.8, 44]} /><Anodized color="#15181d" rough={0.5} />
          </mesh>
        ))}
        <mesh position={[0, 0, 55]} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[R * 0.55, R * 0.55, 3, 40]} /><Anodized color="#0d0f12" /></mesh>
        <RoundedBox args={[R * 0.9, 4, 12]} radius={1} position={[0, R - 8, 30]}><meshStandardMaterial map={plate} roughness={0.4} /></RoundedBox>
        <Clamp position={[0, 0, 26]} axis={[0, 0, 1]} radius={R - 6} width={12} knob={false} />
        {hull && (
          <mesh position={[0, 0, 28]} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[R + 5, R + 5, 66, 48]} /><HullMat color={hull} /></mesh>
        )}
        <MountRig q={q} anchor={mount.anchor} groundY={-source.position_mm[1]} postXZ={mount.post} />
      </group>
      {/* driver box + cable */}
      <group position={[driver.x, driver.y, driver.z]} rotation={[0, driver.ang, 0]}>
        <RoundedBox args={[78, 34, 110]} radius={4} smoothness={4} position={[0, 17, 0]} castShadow receiveShadow>
          <Anodized color="#1c2027" />
        </RoundedBox>
        <mesh position={[0, 17, 55.2]}><boxGeometry args={[70, 26, 0.6]} /><Alu color="#aeb5bf" /></mesh>
        <mesh position={[-20, 20, 56]}><sphereGeometry args={[2.2, 14, 10]} /><meshBasicMaterial color={wavelengthNm == null ? '#444' : glow} toneMapped={false} /></mesh>
        <mesh position={[8, 20, 56]} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[7, 7, 6, 32]} /><Anodized color="#101216" /></mesh>
        {[-1, 1].map((sx) => [-1, 1].map((sz) => <mesh key={`${sx}${sz}`} position={[sx * 30, -1.5, sz * 44]}><cylinderGeometry args={[4, 4.5, 3, 16]} /><Rubber /></mesh>))}
      </group>
      <mesh geometry={cable}><Plastic color="#101114" rough={0.4} /></mesh>
    </group>
  )
}

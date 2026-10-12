/** Shared mechanical parts for all hardware: posts, clamps, bases, joints, selection hull. */
import { useMemo } from 'react'
import * as THREE from 'three'
import { Anodized, Alu, Chrome, Rubber } from './materials'

type V3 = [number, number, number]

/** Cylinder rod between two points (local coordinates of its parent group). */
export function Post({ from, to, radius = 6, material = 'alu', segments = 28 }:
  { from: V3; to: V3; radius?: number; material?: 'alu' | 'anodized' | 'chrome'; segments?: number }) {
  const { pos, quat, len } = useMemo(() => {
    const a = new THREE.Vector3(...from), b = new THREE.Vector3(...to)
    const d = b.clone().sub(a)
    const len = d.length()
    const q = new THREE.Quaternion()
    if (len > 1e-6) q.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.clone().normalize())
    return { pos: a.add(b).multiplyScalar(0.5), quat: q, len }
  }, [from, to])
  if (len < 1e-3) return null
  return (
    <mesh position={pos} quaternion={quat} castShadow receiveShadow>
      <cylinderGeometry args={[radius, radius, len, segments]} />
      {material === 'alu' ? <Alu rough={0.3} /> : material === 'chrome' ? <Chrome /> : <Anodized />}
    </mesh>
  )
}

/** Right-angle / swivel clamp collar with a locking thumb-screw, centred on `position`, axis along `axis`. */
export function Clamp({ position, axis = [0, 1, 0], radius = 8, width = 16, knob = true }:
  { position: V3; axis?: V3; radius?: number; width?: number; knob?: boolean }) {
  const q = useMemo(() => new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...axis).normalize()), [axis])
  return (
    <group position={position} quaternion={q}>
      <mesh castShadow receiveShadow>
        <cylinderGeometry args={[radius + 4, radius + 4, width, 32]} />
        <Anodized color="#101216" rough={0.5} />
      </mesh>
      <mesh position={[0, 0, 0]}>
        <cylinderGeometry args={[radius + 4.4, radius + 4.4, 1.2, 32]} />
        <Chrome />
      </mesh>
      {knob && (
        <group position={[radius + 4, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
          <mesh position={[0, 4, 0]} castShadow>
            <cylinderGeometry args={[1.6, 1.6, 8, 12]} />
            <Chrome />
          </mesh>
          <mesh position={[0, 10, 0]} castShadow>
            <cylinderGeometry args={[5, 5, 5, 24]} />
            <Rubber color="#15171b" />
          </mesh>
        </group>
      )}
    </group>
  )
}

/** Ball joint (articulated arm). */
export function Joint({ position, radius = 7 }: { position: V3; radius?: number }) {
  return (
    <group position={position}>
      <mesh castShadow><sphereGeometry args={[radius, 32, 24]} /><Chrome color="#dfe4ec" /></mesh>
      <mesh position={[0, radius * 0.1, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[radius * 0.98, 0.9, 8, 40]} /><Anodized color="#0b0c0e" />
      </mesh>
    </group>
  )
}

/** Table-standing base plate with a bolt-down slot look; centred on `position` (y = table top = world 0). */
export function Base({ position, radius = 38, thickness = 10 }: { position: V3; radius?: number; thickness?: number }) {
  return (
    <group position={position}>
      <mesh position={[0, thickness / 2, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[radius, radius + 2, thickness, 48]} />
        <Anodized color="#13151a" rough={0.5} />
      </mesh>
      <mesh position={[0, thickness + 0.4, 0]}>
        <cylinderGeometry args={[radius * 0.5, radius * 0.5, 0.8, 32]} />
        <Chrome color="#cfd5de" />
      </mesh>
      {[0, 1, 2].map((i) => {
        const a = (i / 3) * Math.PI * 2 + 0.5
        return (
          <mesh key={i} position={[Math.cos(a) * radius * 0.75, thickness + 0.8, Math.sin(a) * radius * 0.75]}>
            <cylinderGeometry args={[3, 3, 1.6, 6]} />
            <Chrome />
          </mesh>
        )
      })}
    </group>
  )
}

/** Compute a world-axis-aligned mount from a device pose: returns the offset (from the device origin, in world
 * orientation) of a local anchor point, plus the device's height above the table. */
export function useMountOffset(devicePos: V3, q: THREE.Quaternion, anchorLocal: V3) {
  return useMemo(() => {
    const off = new THREE.Vector3(...anchorLocal).applyQuaternion(q)
    return { off: [off.x, off.y, off.z] as V3, groundY: -devicePos[1] }
  }, [devicePos, q, anchorLocal])
}

/** Post + base + clamp + horizontal arm from the table up to an anchor, all in a group that cancels the device
 * rotation so children are specified in world orientation relative to the device origin. */
export function MountRig({ q, anchor, groundY, postXZ, radius = 6.5, armRadius = 5 }:
  { q: THREE.Quaternion; anchor: V3; groundY: number; postXZ?: [number, number]; radius?: number; armRadius?: number }) {
  const inv = useMemo(() => q.clone().invert(), [q])
  const [px, pz] = postXZ ?? [anchor[0], anchor[2]]
  const top = anchor[1]
  const tooLow = top + -groundY < 15
  if (tooLow) return null
  const horizontal = Math.hypot(px - anchor[0], pz - anchor[2]) > 1
  return (
    <group quaternion={inv}>
      <Base position={[px, groundY, pz]} />
      <Post from={[px, groundY + 10, pz]} to={[px, top + 10, pz]} radius={radius} />
      <Clamp position={[px, top, pz]} axis={[0, 1, 0]} radius={radius} width={18} />
      {horizontal && <Post from={[px, top, pz]} to={anchor} radius={armRadius} material="chrome" />}
      <Joint position={anchor} radius={armRadius + 3} />
    </group>
  )
}

/** Selection / hover hull: inverted-hull glow that costs one extra draw per element. */
export function hullColor(selected?: boolean, hovered?: boolean) {
  return selected ? '#4da3ff' : hovered ? '#9fd0ff' : null
}
export function HullMat({ color }: { color: string }) {
  return <meshBasicMaterial color={color} side={THREE.BackSide} transparent opacity={0.85} toneMapped={false} depthWrite={false} />
}

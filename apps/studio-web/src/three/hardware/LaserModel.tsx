import { useMemo } from 'react'
import * as THREE from 'three'
import { Anodized, Alu, Chrome, Rubber, KnurlMetal } from '../util/materials'
import { MountRig, Clamp, HullMat, hullColor } from '../util/parts'
import { labelPlate, warningSticker } from '../util/textures'
import { glowColor, mountFor, outwardXZ, type SourcePartProps } from './sourceCommon'

function ZCyl({ z0, z1, r0, r1 = r0, seg = 48, children, shadow = true }:
  { z0: number; z1: number; r0: number; r1?: number; seg?: number; children?: React.ReactNode; shadow?: boolean }) {
  return (
    <mesh position={[0, 0, (z0 + z1) / 2]} rotation={[Math.PI / 2, 0, 0]} castShadow={shadow} receiveShadow={shadow}>
      <cylinderGeometry args={[r1, r0, z1 - z0, seg]} />{children}
    </mesh>
  )
}

/** Laser head: black cylindrical housing, chrome aperture with a bright emitter, wavelength band, key switch, warning triangle. */
export function LaserModel({ source, wavelengthNm, q, selected, hovered }: SourcePartProps) {
  const R = 18
  const L = 130
  const hull = hullColor(selected, hovered)
  const col = glowColor(wavelengthNm, 3)
  const band = useMemo(() => (wavelengthNm == null ? new THREE.Color('#444a55') : glowColor(wavelengthNm, 0.9)), [wavelengthNm])
  const mount = useMemo(() => mountFor(source, q, [0, R + 5, L * 0.35]), [source, q])
  const plate = labelPlate(['LASER', `${wavelengthNm == null ? '—' : Math.round(wavelengthNm)} nm · CLASS 3B`], { w: 384, h: 128 })
  const cable = useMemo(() => {
    const o = outwardXZ(source)
    const tail = new THREE.Vector3(0, 0, L + 8).applyQuaternion(q)
    const back = new THREE.Vector3(0, 0, 1).applyQuaternion(q)
    const gy = -source.position_mm[1]
    return new THREE.TubeGeometry(new THREE.CatmullRomCurve3([
      tail, tail.clone().add(back.multiplyScalar(30)), new THREE.Vector3(tail.x + o.x * 40, Math.max(tail.y - 70, gy + 12), tail.z + o.y * 40),
      new THREE.Vector3(tail.x + o.x * 120, gy + 6, tail.z + o.y * 120),
    ]), 60, 3.4, 10, false)
  }, [source, q])
  return (
    <group>
      <group quaternion={q}>
        <ZCyl z0={0} z1={5} r0={R + 1.8}><Alu color="#d6dce6" rough={0.45} /></ZCyl>
        <mesh position={[0, 0, -0.05]} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[R - 2.5, R - 2.5, 0.3, 48]} /><meshPhysicalMaterial color="#050608" metalness={0.5} roughness={0.15} clearcoat={1} /></mesh>
        <mesh position={[0, 0, 0.3]} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[3.4, 3.4, 0.6, 28]} /><meshBasicMaterial color={col} toneMapped={false} /></mesh>
        <mesh position={[0, 0, 0.2]} rotation={[0, 0, 0]}><torusGeometry args={[4.6, 0.9, 8, 28]} /><Chrome color="#8c95a3" /></mesh>
        <ZCyl z0={5} z1={L} r0={R}><Anodized color="#101216" /></ZCyl>
        <ZCyl z0={22} z1={34} r0={R + 0.4}><meshStandardMaterial color={band} emissive={band} emissiveIntensity={0.8} roughness={0.5} /></ZCyl>
        <ZCyl z0={36} z1={60} r0={R + 1.4}><KnurlMetal color="#1b1e24" /></ZCyl>
        <ZCyl z0={L} z1={L + 10} r0={R - 1} r1={R - 5}><Alu color="#6a717d" /></ZCyl>
        <ZCyl z0={L + 10} z1={L + 18} r0={5.2}><Rubber /></ZCyl>
        {/* heat-sink rings */}
        {[0, 1, 2, 3, 4].map((i) => <ZCyl key={i} z0={78 + i * 9} z1={82 + i * 9} r0={R + 2.6}><Anodized color="#16191e" rough={0.5} /></ZCyl>)}
        {/* key switch box */}
        <group position={[R + 8, 0, 112]}>
          <mesh castShadow><boxGeometry args={[18, 20, 22]} /><Anodized color="#1a1d23" /></mesh>
          <mesh position={[9.4, 0, 0]} rotation={[0, 0, Math.PI / 2]}><cylinderGeometry args={[6, 6, 1.4, 32]} /><Chrome /></mesh>
          <mesh position={[11, 0, 0]} rotation={[0, 0, Math.PI / 2]}><boxGeometry args={[1, 5, 2.2]} /><Alu color="#d8b45a" /></mesh>
          <mesh position={[9.2, 6, 8]}><sphereGeometry args={[1.5, 12, 8]} /><meshBasicMaterial color={wavelengthNm == null ? '#ff3030' : '#34ff8a'} toneMapped={false} /></mesh>
        </group>
        {/* stickers */}
        <mesh position={[0, R + 0.1, 72]} rotation={[-Math.PI / 2, 0, 0]}><planeGeometry args={[19, 19]} /><meshBasicMaterial map={warningSticker('laser')} transparent /></mesh>
        <mesh position={[0, R + 0.08, 100]} rotation={[-Math.PI / 2, 0, 0]}><planeGeometry args={[27, 9]} /><meshStandardMaterial map={plate} roughness={0.4} /></mesh>
        <Clamp position={[0, 0, L * 0.35]} axis={[0, 0, 1]} radius={R + 0.5} width={13} />
        {hull && <ZCyl z0={-2.5} z1={L + 20} r0={R + 6} shadow={false}><HullMat color={hull} /></ZCyl>}
        <MountRig q={q} anchor={mount.anchor} groundY={-source.position_mm[1]} postXZ={mount.post} radius={6} armRadius={4.5} />
      </group>
      <mesh geometry={cable}><meshPhysicalMaterial color="#101114" roughness={0.4} clearcoat={0.5} /></mesh>
    </group>
  )
}

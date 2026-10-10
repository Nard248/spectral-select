import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { RoundedBox } from '@react-three/drei'
import * as THREE from 'three'
import type { CameraModelProps } from '../types'
import { lookAtQuaternion } from '../pose'
import { Anodized, Alu, Chrome, Rubber, KnurlMetal } from '../util/materials'
import { MountRig, HullMat, hullColor } from '../util/parts'
import { labelPlate, scaleRing } from '../util/textures'
import { CameraFrustum } from './CameraFrustum'

type V3 = [number, number, number]

/** cylinder with its axis on local Z, spanning z0..z1 (z grows toward the camera's back; optical axis is −Z) */
function ZCyl({ z0, z1, r0, r1 = r0, seg = 48, children, ...rest }:
  { z0: number; z1: number; r0: number; r1?: number; seg?: number; children?: React.ReactNode; castShadow?: boolean; receiveShadow?: boolean; open?: boolean }) {
  // CylinderGeometry(rTop,rBottom): after rotating +90° about X, top (+Y) points to +Z ⇒ rTop is the z1 end.
  return (
    <mesh position={[0, 0, (z0 + z1) / 2]} rotation={[Math.PI / 2, 0, 0]} castShadow={rest.castShadow ?? true} receiveShadow={rest.receiveShadow ?? true}>
      <cylinderGeometry args={[r1, r0, z1 - z0, seg, 1, rest.open]} />
      {children}
    </mesh>
  )
}

/** Fraction (0..1) along the printed scale for the focus ring. */
function focusFraction(dMm: number) {
  const marks = [300, 500, 1000, 2000, 5000]
  if (dMm <= marks[0]) return 0
  if (dMm >= marks[4]) return 0.8 + 0.2 * Math.min(1, Math.log(dMm / 5000) / Math.log(4))
  for (let i = 0; i < 4; i++) if (dMm <= marks[i + 1]) {
    return (i + (Math.log(dMm / marks[i]) / Math.log(marks[i + 1] / marks[i]))) / 5
  }
  return 1
}

/** Ring rotation about its axis that brings the scale position m∈[0,1] (printed marks span u = 90…890 of 1024) under the
 *  fixed index at local +Y. A cylinder's u runs from local −Y (u = 0) towards +X; the index sits at u = 0.5. */
function ringAngle(m: number) { return Math.PI - 2 * Math.PI * ((90 + 800 * m) / 1024) }

const ANCHOR_X = 49
export function cameraDimensions(camera: CameraModelProps['camera']) {
  const barrel = 38 + 0.55 * camera.lens.focal_length_mm            // lens barrel length (mm) grows with focal length
  const lctf = 54
  const bodyZ0 = barrel + 8 + lctf
  return { barrel, lctf, bodyZ0, bodyLen: 140, total: bodyZ0 + 140 }
}

/** Nuance/LCTF-class hyperspectral camera: body + LCTF stack + C-mount lens (knurled focus/aperture rings
 *  that rotate with `focus_distance_mm` / `f_number`), mount post + clamp to the table, optional frustum overlay. */
export function CameraModel({ camera, selected, hovered, showFrustum, acquiring }: CameraModelProps) {
  const q = useMemo(() => lookAtQuaternion(camera.position_mm, camera.target_mm, camera.roll_deg),
    [camera.position_mm, camera.target_mm, camera.roll_deg])
  const d = cameraDimensions(camera)
  const { barrel, bodyZ0 } = d
  const lctfZ0 = barrel + 8

  const apertureAng = useMemo(() => {
    const m = THREE.MathUtils.clamp(Math.log2(camera.lens.f_number / 1.4) / Math.log2(22 / 1.4), 0, 1)
    return ringAngle(m)
  }, [camera.lens.f_number])
  const focusAng = useMemo(() => ringAngle(focusFraction(camera.lens.focus_distance_mm)), [camera.lens.focus_distance_mm])

  const ledRef = useRef<THREE.MeshBasicMaterial>(null)
  const plates = useRef<(THREE.MeshPhysicalMaterial | null)[]>([])
  const rings = useRef<(THREE.Mesh | null)[]>([])
  const nPlates = 6
  useFrame(({ clock }) => {
    const t = clock.elapsedTime
    if (ledRef.current) {
      const c = acquiring ? new THREE.Color('#ff7a1f').multiplyScalar(1.2 + 1.8 * (0.5 + 0.5 * Math.sin(t * 8)))
        : new THREE.Color('#34ff8a').multiplyScalar(1.4)
      ledRef.current.color.copy(c)
    }
    plates.current.forEach((m, i) => {
      if (!m) return
      m.iridescenceThicknessRange = acquiring ? [180 + 80 * Math.sin(t * 4 + i), 520 + 260 * Math.sin(t * 3 + i * 1.3)] : [250, 650]
      m.opacity = acquiring ? 0.5 + 0.18 * Math.sin(t * 6 + i) : 0.4
    })
    const active = Math.floor(t * 7) % nPlates
    rings.current.forEach((m, i) => { if (m) m.visible = !!acquiring && i === active })
  })

  const bodyMid = bodyZ0 + 70
  const mount = useMemo(() => {
    const a = new THREE.Vector3(ANCHOR_X + 4, 0, bodyMid).applyQuaternion(q)
    const out = new THREE.Vector2(a.x, a.z)
    if (out.length() < 1) out.set(1, 0)
    out.normalize().multiplyScalar(44)               // post stands 44 mm beyond the mount face, on the table
    return { anchor: [a.x, a.y, a.z] as V3, post: [a.x + out.x, a.z + out.y] as [number, number] }
  }, [q, bodyMid])
  const hull = hullColor(selected, hovered)
  const label = labelPlate(['SpectralForge HSI', 'LCTF  ·  400 – 720 nm  ·  CCD', 'SN SF-HSI-0042'])
  const ringMapA = scaleRing('aperture'), ringMapF = scaleRing('focus')

    const fins = useMemo(() => Array.from({ length: 11 }, (_, i) => -40 + i * 8), [])
  const cable = useMemo(() => {
    const zr = d.total
    return new THREE.CatmullRomCurve3([
      new THREE.Vector3(22, -22, zr - 2), new THREE.Vector3(22, -22, zr + 26), new THREE.Vector3(34, -30, zr + 62),
      new THREE.Vector3(60, -34, zr + 84), new THREE.Vector3(96, -34, zr + 94),
    ])
  }, [d.total])
  const lctfCable = useMemo(() => new THREE.CatmullRomCurve3([
    new THREE.Vector3(38, 0, lctfZ0 + 27), new THREE.Vector3(44, 0, lctfZ0 + 38), new THREE.Vector3(40, 10, bodyZ0 + 3), new THREE.Vector3(28, 36, bodyZ0 + 1),
  ]), [lctfZ0, bodyZ0])

  return (
    <group position={camera.position_mm} quaternion={q}>
      {/* ---------------------------------------------------------------- lens */}
      <group>
        {/* hood / filter-thread ring */}
        <ZCyl z0={-7} z1={3} r0={33.5} r1={33.5}><KnurlMetal color="#1f232a" /></ZCyl>
        <ZCyl z0={-7.2} z1={-6.2} r0={33.8}><Chrome color="#9aa3af" /></ZCyl>
        {/* front glass, coated, recessed */}
        <mesh position={[0, 0, -2]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[27.5, 27.5, 1.2, 48]} />
          <meshPhysicalMaterial color="#0a1018" metalness={0.1} roughness={0.04} clearcoat={1} clearcoatRoughness={0.02}
            iridescence={1} iridescenceIOR={1.55} iridescenceThicknessRange={[200, 420]} envMapIntensity={2.2} />
        </mesh>
        {[1, 2, 3].map((i) => (
          <mesh key={i} position={[0, 0, -2 + i * 5]} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[26 - i * 3.2, 0.9, 10, 48]} />
            <Chrome color="#556070" />
          </mesh>
        ))}
        {/* fixed barrel */}
        <ZCyl z0={3} z1={barrel} r0={31} r1={31}><Anodized color="#2a2f38" /></ZCyl>
        {/* focus ring */}
        <group rotation={[0, 0, focusAng]}>
          <ZCyl z0={9} z1={9 + barrel * 0.4} r0={33.2}><KnurlMetal map={ringMapF} /></ZCyl>
        </group>
        <ZCyl z0={9 + barrel * 0.4} z1={12 + barrel * 0.4} r0={32.2}><Chrome color="#8e97a5" /></ZCyl>
        {/* aperture ring */}
        <group rotation={[0, 0, apertureAng]}>
          <ZCyl z0={12 + barrel * 0.4} z1={12 + barrel * 0.4 + 15} r0={32.6}><KnurlMetal map={ringMapA} /></ZCyl>
        </group>
        <ZCyl z0={barrel - 7} z1={barrel} r0={32}><Anodized color="#1b1e25" /></ZCyl>
        {/* index marks */}
        <mesh position={[0, 33.6, 9 + barrel * 0.2]}><boxGeometry args={[1.6, 1, barrel * 0.4]} /><meshBasicMaterial color="#f5f7fa" /></mesh>
        <mesh position={[0, 33.6, 12 + barrel * 0.4 + 7.5]}><boxGeometry args={[1.8, 1, 15]} /><meshBasicMaterial color="#ffb22e" /></mesh>
        {/* C-mount adapter */}
        <ZCyl z0={barrel} z1={lctfZ0} r0={25}><Alu /></ZCyl>
        <ZCyl z0={barrel + 1.5} z1={barrel + 2.4} r0={29}><Chrome color="#c4cbd6" /></ZCyl>
      </group>

      {/* ---------------------------------------------------------------- LCTF module */}
      <group>
        {[lctfZ0, lctfZ0 + d.lctf - 5].map((z, i) => (
          <ZCyl key={i} z0={z} z1={z + 5} r0={37}><Anodized color="#3a404c" rough={0.4} /></ZCyl>
        ))}
        {[0, 1, 2, 3].map((i) => {
          const a = (i / 4) * Math.PI * 2 + Math.PI / 4
          return <mesh key={i} position={[Math.cos(a) * 32, Math.sin(a) * 32, lctfZ0 + d.lctf / 2]} rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[2.2, 2.2, d.lctf - 4, 14]} /><Chrome color="#d5dbe4" />
          </mesh>
        })}
        {Array.from({ length: nPlates }, (_, i) => {
          const z = lctfZ0 + 10 + i * ((d.lctf - 20) / (nPlates - 1))
          return (
            <group key={i}>
              <mesh position={[0, 0, z]} rotation={[Math.PI / 2, 0, 0]}>
                <cylinderGeometry args={[27, 27, 1.1, 64]} />
                <meshPhysicalMaterial ref={(m) => { plates.current[i] = m }} color="#cfe6ff" transparent opacity={0.4} metalness={0.15} roughness={0.04}
                  iridescence={1} iridescenceIOR={1.9} iridescenceThicknessRange={[250, 650]} side={THREE.DoubleSide} depthWrite={false}
                  envMapIntensity={1.6} clearcoat={1} />
              </mesh>
              <mesh ref={(m) => { rings.current[i] = m }} position={[0, 0, z]} rotation={[0, 0, 0]} visible={false}>
                <torusGeometry args={[27.6, 0.6, 10, 64]} />
                <meshBasicMaterial color={new THREE.Color('#6fe8ff').multiplyScalar(2.4)} toneMapped={false} />
              </mesh>
            </group>
          )
        })}
        {/* LC driver board */}
        <mesh position={[41, 0, lctfZ0 + d.lctf / 2]}>
          <boxGeometry args={[4, 24, 38]} /><meshPhysicalMaterial color="#0f3d2e" roughness={0.45} metalness={0.2} clearcoat={0.6} />
        </mesh>
        {[-10, -3, 4, 11].map((z, i) => (
          <mesh key={i} position={[43.6, 6 - i * 4, lctfZ0 + d.lctf / 2 + z]}><boxGeometry args={[1.4, 3.5, 5]} /><Anodized color="#050607" /></mesh>
        ))}
        <mesh><tubeGeometry args={[lctfCable, 24, 1.8, 8, false]} /><Rubber /></mesh>
      </group>

      {/* ---------------------------------------------------------------- body */}
      <group>
        <RoundedBox args={[90, 90, d.bodyLen]} radius={7} smoothness={5} bevelSegments={4} position={[0, 0, bodyMid]} castShadow receiveShadow>
          <Anodized color="#3a4150" rough={0.42} />
        </RoundedBox>
        {/* front flange */}
        <RoundedBox args={[96, 96, 7]} radius={3.5} smoothness={4} position={[0, 0, bodyZ0 + 3.5]} castShadow>
          <Anodized color="#434b5a" rough={0.34} />
        </RoundedBox>
        {/* cooling fins on ±Y */}
        {[1, -1].map((sgn) => fins.map((x, i) => (
          <mesh key={`${sgn}${i}`} position={[x, sgn * 51.5, bodyZ0 + 106]} castShadow>
            <boxGeometry args={[2.6, 13, 62]} /><Anodized color="#272c35" rough={0.5} />
          </mesh>
        )))}
        {[1, -1].map((sgn) => (
          <mesh key={sgn} position={[0, sgn * 46.2, bodyZ0 + 106]}>
            <boxGeometry args={[92, 2, 64]} /><Anodized color="#1f232a" />
          </mesh>
        ))}
        {/* tripod plate / mount face on +X */}
        <mesh position={[ANCHOR_X - 2.5, 0, bodyMid]} castShadow><boxGeometry args={[7, 46, 60]} /><Alu rough={0.3} /></mesh>
        {[-18, 18].map((z) => <mesh key={z} position={[ANCHOR_X + 1.2, 0, bodyMid + z]} rotation={[0, 0, Math.PI / 2]}><cylinderGeometry args={[3, 3, 1.4, 6]} /><Chrome /></mesh>)}
        {/* label plates on ±Y (front half) + trim lines */}
        {[1, -1].map((sgn) => (
          <group key={sgn}>
            <mesh position={[0, sgn * 45.12, bodyZ0 + 40]} rotation={[sgn * -Math.PI / 2, 0, sgn > 0 ? Math.PI / 2 : -Math.PI / 2]}>
              <planeGeometry args={[62, 15.5]} />
              <meshStandardMaterial map={label} roughness={0.35} metalness={0.2} />
            </mesh>
            {[-1, 1].map((sx) => <mesh key={sx} position={[sx * 44.7, sgn * 44.7, bodyMid]}><boxGeometry args={[1.2, 1.2, d.bodyLen - 6]} /><Chrome color="#aab3c2" /></mesh>)}
            <mesh position={[0, sgn * 45.5, bodyZ0 + 74]}><boxGeometry args={[84, 0.8, 1.2]} /><meshBasicMaterial color={new THREE.Color('#38d0ff').multiplyScalar(1.6)} toneMapped={false} /></mesh>
          </group>
        ))}
        {/* status LED + lightpipe */}
        <mesh position={[38, 38, bodyZ0 - 0.4]} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[3.2, 3.2, 1, 24]} /><Chrome color="#555d6a" /></mesh>
        <mesh position={[38, 38, bodyZ0 - 1]}><sphereGeometry args={[2.3, 20, 14]} /><meshBasicMaterial ref={ledRef} color="#34ff8a" toneMapped={false} /></mesh>
        {/* rear panel */}
        <group position={[0, 0, d.total]}>
          <mesh position={[0, 0, 0.3]}><boxGeometry args={[80, 80, 0.8]} /><Anodized color="#1a1d23" rough={0.55} /></mesh>
          <mesh position={[-22, 14, 1.5]}><boxGeometry args={[14, 6.5, 3]} /><meshStandardMaterial color="#1f6bff" roughness={0.4} /></mesh>
          <mesh position={[-22, 14, 3.2]}><boxGeometry args={[11, 3.8, 0.8]} /><Rubber color="#000" /></mesh>
          <mesh position={[-22, -2, 1.8]}><boxGeometry args={[18, 8, 3.6]} /><Chrome color="#b9c0cc" /></mesh>
          {[-4, 14, 30].map((x, i) => (
            <group key={i} position={[x, -22, 1]} rotation={[Math.PI / 2, 0, 0]}>
              <mesh><cylinderGeometry args={[i === 0 ? 5.5 : 4.2, i === 0 ? 5.5 : 4.2, 5, 24]} /><Chrome /></mesh>
              <mesh position={[0, 2.6, 0]}><cylinderGeometry args={[i === 0 ? 2.2 : 1.6, i === 0 ? 2.2 : 1.6, 1, 14]} /><Alu color="#e9d6a0" /></mesh>
            </group>
          ))}
          {[0, 1, 2, 3].map((i) => <mesh key={i} position={[22 + i * 6, 14, 1.4]}><boxGeometry args={[4, 4, 2]} /><meshBasicMaterial color={['#3dff8a', '#ffd23d', '#3dff8a', '#222'][i]} toneMapped={false} /></mesh>)}
        </group>
        <mesh><tubeGeometry args={[cable, 40, 4.4, 12, false]} /><Rubber color="#0a0b0d" /></mesh>
        <group position={[96, -34, d.total + 94]}>
          <mesh rotation={[0, 0, Math.PI / 2]}><cylinderGeometry args={[7, 6, 22, 24]} /><Anodized color="#1d2026" /></mesh>
          <mesh position={[13, 0, 0]} rotation={[0, 0, Math.PI / 2]}><cylinderGeometry args={[3, 3, 6, 16]} /><Rubber /></mesh>
        </group>
      </group>

      {/* selection hull */}
      {hull && (
        <group>
          <mesh position={[0, 0, bodyMid]}><boxGeometry args={[99, 99, d.bodyLen + 8]} /><HullMat color={hull} /></mesh>
          <mesh position={[0, 0, (barrel + lctfZ0 + d.lctf) / 2 - 1]} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[38.5, 38.5, barrel + 8 + d.lctf + 10, 40]} /><HullMat color={hull} /></mesh>
        </group>
      )}

      {/* world-axis mount: vertical post + clamp + arm to the body's +X mount face */}
      <MountRig q={q} anchor={mount.anchor} groundY={-camera.position_mm[1]} postXZ={mount.post} />

      {showFrustum && <CameraFrustum camera={camera} acquiring={acquiring} />}
    </group>
  )
}

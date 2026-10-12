import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { RoundedBox } from '@react-three/drei'
import * as THREE from 'three'
import { Anodized, Alu, Chrome, Gold, Rubber, Plastic, KnurlMetal } from '../util/materials'
import { MountRig, Clamp, HullMat, hullColor } from '../util/parts'
import { brushedRoughness, drawLcd, labelPlate, warningSticker } from '../util/textures'
import { glowColor, mountFor, outwardXZ, type SourcePartProps, type V3 } from './sourceCommon'

function ZCyl({ z0, z1, r0, r1 = r0, seg = 48, children, shadow = true }:
  { z0: number; z1: number; r0: number; r1?: number; seg?: number; children?: React.ReactNode; shadow?: boolean }) {
  return (
    <mesh position={[0, 0, (z0 + z1) / 2]} rotation={[Math.PI / 2, 0, 0]} castShadow={shadow} receiveShadow={shadow}>
      <cylinderGeometry args={[r1, r0, z1 - z0, seg]} />
      {children}
    </mesh>
  )
}

const UNIT = { W: 190, H: 118, D: 236, foot: 7 }

/** Benchtop xenon lamp + monochromator: glowing LCD ("365 nm"), knobs that turn with wavelength, vents, UV sticker. */
function XenonUnit({ wavelengthNm, fwhm, name, glow, hull }: { wavelengthNm: number | null; fwhm: number; name: string; glow: THREE.Color; hull: string | null }) {
  const { W, H, D, foot } = UNIT
  const lcd = useRef<THREE.CanvasTexture | null>(null)
  const tex = useMemo(() => drawLcd(null, wavelengthNm, fwhm, name), []) // eslint-disable-line react-hooks/exhaustive-deps
  lcd.current = tex
  useEffect(() => { drawLcd(tex, wavelengthNm, fwhm, name) }, [tex, wavelengthNm, fwhm, name])
  const knobAng = wavelengthNm == null ? -2.3 : THREE.MathUtils.mapLinear(THREE.MathUtils.clamp(wavelengthNm, 250, 800), 250, 800, -2.3, 2.3)
  const warm = useMemo(() => new THREE.MeshBasicMaterial({ color: '#ff8a2a', toneMapped: false }), [])
  useFrame(({ clock }) => {
    const k = wavelengthNm == null ? 0.25 : 1.1 + 0.06 * Math.sin(clock.elapsedTime * 11)
    warm.color.setRGB(1.0 * k, 0.5 * k, 0.16 * k)
  })
  const plate = labelPlate(['SpectralForge TLS-300', 'Xenon 300 W · Monochromator · 250 – 800 nm'], { w: 768, h: 128 })
  const portY = H * 0.36
  return (
    <group position={[0, foot, 0]}>
      {/* chassis */}
      <RoundedBox args={[W, H, D]} radius={5} smoothness={4} position={[0, H / 2, 0]} castShadow receiveShadow>
        <meshPhysicalMaterial color="#14161b" metalness={0.5} roughness={0.55} roughnessMap={undefined} clearcoat={0.3} />
      </RoundedBox>
      {/* front panel */}
      <mesh position={[0, H / 2, D / 2 + 0.4]} castShadow>
        <boxGeometry args={[W - 6, H - 8, 2.4]} /><meshPhysicalMaterial color="#d5dae2" metalness={0.5} roughness={0.5} roughnessMap={brushedRoughness()} />
      </mesh>
      <mesh position={[0, H * 0.14, D / 2 + 1.7]}><boxGeometry args={[W - 14, 0.8, 0.4]} /><meshBasicMaterial color="#1b1f26" /></mesh>
      {/* LCD */}
      <group position={[-W * 0.17, H * 0.66, D / 2 + 1.7]}>
        <mesh><boxGeometry args={[80, 29, 1.2]} /><Plastic color="#06070a" rough={0.3} /></mesh>
        <mesh position={[0, 0, 0.66]}><planeGeometry args={[74, 23.1]} /><meshBasicMaterial map={tex} toneMapped={false} /></mesh>
      </group>
      {/* wavelength knob (big) */}
      <group position={[W * 0.255, H * 0.62, D / 2 + 1.6]} rotation={[Math.PI / 2, 0, 0]}>
        <mesh position={[0, -0.8, 0]}><cylinderGeometry args={[22, 22, 1, 48]} /><Chrome color="#566070" /></mesh>
        {Array.from({ length: 23 }, (_, i) => {
          const a = THREE.MathUtils.mapLinear(i, 0, 22, -2.3, 2.3)
          return <mesh key={i} position={[Math.sin(a) * 20.5, 0.1, -Math.cos(a) * 20.5]} rotation={[0, a, 0]}><boxGeometry args={[0.7, 0.4, i % 5 === 0 ? 3.2 : 1.6]} /><meshBasicMaterial color="#f1f4f8" /></mesh>
        })}
        <group rotation={[0, -knobAng, 0]}>
          <mesh position={[0, 7.5, 0]} castShadow><cylinderGeometry args={[14.5, 16.5, 14, 56]} /><KnurlMetal color="#16181d" /></mesh>
          <mesh position={[0, 14.6, 0]}><cylinderGeometry args={[12.5, 14, 1.4, 56]} /><Alu color="#2c313a" rough={0.2} /></mesh>
          <mesh position={[0, 15.4, -8]}><boxGeometry args={[1.6, 0.4, 8]} /><meshBasicMaterial color="#ffb22e" /></mesh>
        </group>
      </group>
      {/* two small knobs */}
      {[-1, 1].map((s) => (
        <group key={s} position={[W * (0.255 + s * 0.15) - 0, H * 0.22, D / 2 + 1.6]} rotation={[Math.PI / 2, 0, 0]}>
          <mesh position={[0, 4, 0]} castShadow><cylinderGeometry args={[7, 8, 8, 40]} /><KnurlMetal color="#16181d" /></mesh>
          <mesh position={[0, 8.3, 0]}><boxGeometry args={[1, 0.4, 5]} /><meshBasicMaterial color="#f1f4f8" /></mesh>
        </group>
      ))}
      {/* output coupler */}
      <group position={[-W * 0.28, portY - 8, D / 2 + 1.6]} rotation={[Math.PI / 2, 0, 0]}>
        <mesh position={[0, 3, 0]} castShadow><cylinderGeometry args={[14, 14, 6, 48]} /><Chrome /></mesh>
        <mesh position={[0, 6.6, 0]}><cylinderGeometry args={[9, 9, 1.2, 40]} /><Gold /></mesh>
        <mesh position={[0, 7.3, 0]}><cylinderGeometry args={[5.5, 5.5, 0.6, 40]} /><meshBasicMaterial color={glow} toneMapped={false} /></mesh>
      </group>
      {/* power switch + LEDs */}
      <mesh position={[-W * 0.4, H * 0.22, D / 2 + 2.4]} castShadow><boxGeometry args={[11, 15, 3.2]} /><Plastic color="#0c0d10" /></mesh>
      <mesh position={[-W * 0.4, H * 0.22 + 2, D / 2 + 4.3]} rotation={[-0.35, 0, 0]}><boxGeometry args={[9, 10, 1]} /><meshBasicMaterial color="#e9edf2" /></mesh>
      {[0, 1, 2].map((i) => (
        <mesh key={i} position={[-W * 0.4 + 18 + i * 8, H * 0.22, D / 2 + 1.9]}><sphereGeometry args={[1.8, 14, 10]} />
          <meshBasicMaterial color={[new THREE.Color('#34ff8a').multiplyScalar(2), glow.clone(), new THREE.Color('#ffb02e').multiplyScalar(wavelengthNm == null ? 0.2 : 2)][i]} toneMapped={false} /></mesh>
      ))}
      <mesh position={[0, H * 0.075, D / 2 + 1.8]}><planeGeometry args={[W * 0.7, 13]} /><meshStandardMaterial map={plate} roughness={0.4} metalness={0.2} /></mesh>
      {/* top vents: glowing lamp through slots */}
      <group position={[0, H + 0.05, -20]}>
        <mesh rotation={[-Math.PI / 2, 0, 0]}><planeGeometry args={[W - 40, 120]} /><Anodized color="#08090b" /></mesh>
        {Array.from({ length: 18 }, (_, i) => (
          <mesh key={i} position={[0, 0.25, -56 + i * 6.6]} rotation={[-Math.PI / 2, 0, 0]} material={warm}><planeGeometry args={[W - 56, 2.2]} /></mesh>
        ))}
      </group>
      {/* UV sticker + rear fan */}
      <mesh position={[W / 2 - 24, H + 0.1, 70]} rotation={[-Math.PI / 2, 0, 0]}><planeGeometry args={[26, 26]} /><meshBasicMaterial map={warningSticker('uv')} transparent /></mesh>
      <group position={[0, H / 2, -D / 2 - 0.2]} rotation={[-Math.PI / 2, 0, 0]}>
        {[22, 17, 12, 7].map((r) => <mesh key={r} position={[0, -0.6 * (22 - r) / 5, 0]}><torusGeometry args={[r, 0.7, 6, 40]} /><Alu color="#3a3f48" /></mesh>)}
        <mesh position={[0, 0.2, 0]}><cylinderGeometry args={[4, 4, 1, 20]} /><Anodized color="#050607" /></mesh>
      </group>
      {/* side louvres + badge */}
      {[-1, 1].map((sx) => (
        <group key={sx} position={[sx * (W / 2 + 0.15), H / 2, -20]} rotation={[0, sx * Math.PI / 2, 0]}>
          {Array.from({ length: 9 }, (_, i) => (
            <mesh key={i} position={[0, -30 + i * 7.5, 0]}><boxGeometry args={[130, 2.6, 0.6]} /><Anodized color="#030304" /></mesh>
          ))}
          <mesh position={[0, 40, 0]}><boxGeometry args={[130, 0.6, 0.8]} /><Alu color="#8d95a1" /></mesh>
        </group>
      ))}
      {/* feet */}
      {[[-1, -1], [1, -1], [-1, 1], [1, 1]].map(([x, z], i) => (
        <mesh key={i} position={[x * (W / 2 - 14), -foot / 2, z * (D / 2 - 18)]}><cylinderGeometry args={[7, 8, foot, 24]} /><Rubber /></mesh>
      ))}
      {hull && <mesh position={[0, H / 2, 0]}><boxGeometry args={[W + 6, H + 6, D + 6]} /><HullMat color={hull} /></mesh>}
    </group>
  )
}

/** Tunable lamp: collimator head on an articulated post (exit at source.position, beam −Z) + fibre-optic cable + benchtop unit. */
export function TunableLampModel({ source, wavelengthNm, q, selected, hovered }: SourcePartProps) {
  const D = source.beam.aperture_diameter_mm
  const R = Math.max(13, D / 2 + 7)
  const L = 66
  const hull = hullColor(selected, hovered)
  const glow = glowColor(wavelengthNm)
  const mount = useMemo(() => mountFor(source, q, [0, R + 6, L * 0.5]), [source, q, R])

  // ---- world-aligned accessories: unit + fibre
  const acc = useMemo(() => {
    const o = outwardXZ(source)
    const p = source.position_mm
    const toCentre = new THREE.Vector2(-p[0], -p[2])
    const perp = new THREE.Vector2(-o.y, o.x)
    if (perp.dot(toCentre) < 0) perp.multiplyScalar(-1)
    const cx = o.x * 20 + perp.x * (UNIT.W / 2 + 70), cz = o.y * 20 + perp.y * (UNIT.W / 2 + 70)
    // operator side: the control panel faces world +Z (towards the viewer in the default studio view)
    const yaw = 0
    const frontLocal = new THREE.Vector3(-UNIT.W * 0.28, UNIT.foot + UNIT.H * 0.36 - 8, UNIT.D / 2 + 12)
    const unitQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw)
    const port = frontLocal.clone().applyQuaternion(unitQ).add(new THREE.Vector3(cx, -p[1], cz))
    const front = new THREE.Vector3(0, 0, 1).applyQuaternion(unitQ)
    const tail = new THREE.Vector3(0, 0, L + 16).applyQuaternion(q)
    const back = new THREE.Vector3(0, 0, 1).applyQuaternion(q)
    const mid = port.clone().lerp(tail, 0.5); mid.y = Math.max(port.y, tail.y) * 0.5 + 12
    const curve = new THREE.CatmullRomCurve3([
      port, port.clone().add(front.clone().multiplyScalar(26)).add(new THREE.Vector3(0, 12, 0)),
      port.clone().lerp(mid, 0.55).setY(port.y + 6),
      mid.clone().setY(Math.max(mid.y, -p[1] + 22)),
      tail.clone().add(back.clone().multiplyScalar(110)).add(new THREE.Vector3(0, -30, 0)).setY(Math.max(tail.y - 30, -p[1] + 20)),
      tail.clone().add(back.clone().multiplyScalar(36)),
      tail,
    ], false, 'catmullrom', 0.5)
    return { cx, cz, yaw, curve, y: -p[1] }
  }, [source, q])
  const tube = useMemo(() => new THREE.TubeGeometry(acc.curve, 120, 3.1, 12, false), [acc])
  useEffect(() => () => tube.dispose(), [tube])

  return (
    <group>
      <group quaternion={q}>
        {/* ---- collimator head (exit aperture at origin) */}
        <ZCyl z0={0} z1={3.2} r0={R + 1.6}><Chrome color="#c2c9d4" /></ZCyl>
        <mesh position={[0, 0, 0.5]} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[D / 2, D / 2, 0.8, 56]} />
          <meshBasicMaterial color={glow} toneMapped={false} /></mesh>
        <mesh position={[0, 0, 0.2]} rotation={[Math.PI / 2, 0, 0]}><torusGeometry args={[D / 2 + 0.6, 0.7, 8, 56]} /><Chrome color="#8c95a3" /></mesh>
        <ZCyl z0={3.2} z1={42} r0={R}><Anodized color="#15181d" /></ZCyl>
        <ZCyl z0={14} z1={30} r0={R + 1.7}><KnurlMetal color="#1d2127" /></ZCyl>
        <ZCyl z0={30} z1={33} r0={R + 0.8}><Chrome color="#7d8695" /></ZCyl>
        <ZCyl z0={42} z1={58} r0={R} r1={8.5}><Anodized color="#101216" /></ZCyl>
        <ZCyl z0={58} z1={L} r0={7}><Gold /></ZCyl>
        <ZCyl z0={L} z1={L + 16} r0={3.8} r1={5.2}><Rubber /></ZCyl>
        <Clamp position={[0, 0, L * 0.5]} axis={[0, 0, 1]} radius={R + 1} width={13} />
        {hull && <ZCyl z0={-2.5} z1={L + 18} r0={R + 5} shadow={false}><HullMat color={hull} /></ZCyl>}
        <MountRig q={q} anchor={mount.anchor} groundY={-source.position_mm[1]} postXZ={mount.post} radius={6} armRadius={4.5} />
      </group>

      {/* ---- fibre + benchtop unit (world orientation) */}
      <mesh geometry={tube} castShadow><meshPhysicalMaterial color="#14161a" roughness={0.38} metalness={0.15} clearcoat={0.6} clearcoatRoughness={0.3} /></mesh>
      <group position={[acc.cx, acc.y, acc.cz]} rotation={[0, acc.yaw, 0]}>
        <XenonUnit wavelengthNm={wavelengthNm} fwhm={source.line_fwhm_nm} name="TLS-300 XE-MONO" glow={glow} hull={hull} />
      </group>
    </group>
  )
}
export type { V3 }

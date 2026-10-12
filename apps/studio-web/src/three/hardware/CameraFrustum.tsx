import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import type { CameraSpec } from '../../generated/experiment'

/** Thin-lens imaging geometry for the overlay (all mm). Mirrors the engine's optics model:
 *  image distance v = f·s/(s−f); half field at the focus plane = (sensor/2)·(s−f)/f.
 *  Depth of field from circle of confusion c = 2 px·pitch·binning: H = f²/(N·c)+f,
 *  near = s(H−f)/(H+s−2f), far = s(H−f)/(H−s) (∞ when s ≥ H). */
export function imagingGeometry(cam: CameraSpec) {
  const f = cam.lens.focal_length_mm, s = Math.max(cam.lens.focus_distance_mm, f * 1.05), N = cam.lens.f_number
  const px = (cam.sensor.pixel_pitch_um * cam.sensor.binning) / 1000
  const sw = cam.sensor.width_px * px, sh = cam.sensor.height_px * px
  const v = (f * s) / (s - f)
  const tanX = sw / 2 / v, tanY = sh / 2 / v      // chief-ray slopes through the lens centre
  const c = 2 * px
  const H = (f * f) / (N * c) + f
  const near = (s * (H - f)) / (H + s - 2 * f)
  const far = s >= H ? Infinity : (s * (H - f)) / (H - s)
  return { f, s, v, sw, sh, tanX, tanY, near, far, H, c }
}

const FRUSTUM = new THREE.Color('#56d4ff'), FOCUS = new THREE.Color('#7dffb2'), DOF = new THREE.Color('#ffc857')

function quadCorners(d: number, tx: number, ty: number): THREE.Vector3[] {
  const x = d * tx, y = d * ty
  return [new THREE.Vector3(-x, -y, -d), new THREE.Vector3(x, -y, -d), new THREE.Vector3(x, y, -d), new THREE.Vector3(-x, y, -d)]
}

/** Frustum from the sensor through the lens, translucent focus plane, and depth-of-field slab. */
export function CameraFrustum({ camera, acquiring }: { camera: CameraSpec; acquiring?: boolean }) {
  const g = useMemo(() => imagingGeometry(camera), [camera])
  const farD = Math.min(g.far, g.s * 1.9)
  const matFocus = useRef<THREE.MeshBasicMaterial>(null)

  const { pyramid, edges, dof, focusEdges } = useMemo(() => {
    const L = Math.max(g.s * 1.25, farD)      // pyramid reaches past the focus plane
    const e = quadCorners(L, g.tanX, g.tanY)
    const apex = new THREE.Vector3(0, 0, 0)
    const pos: number[] = []
    for (let i = 0; i < 4; i++) { const a = e[i], b = e[(i + 1) % 4]; pos.push(...apex.toArray(), ...a.toArray(), ...b.toArray()) }
    const pyramid = new THREE.BufferGeometry(); pyramid.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
    const lines: number[] = []
    e.forEach((p) => lines.push(0, 0, 0, p.x, p.y, p.z))
    for (let i = 0; i < 4; i++) { const a = e[i], b = e[(i + 1) % 4]; lines.push(...a.toArray(), ...b.toArray()) }
    const edges = new THREE.BufferGeometry(); edges.setAttribute('position', new THREE.Float32BufferAttribute(lines, 3))
    // focus plane outline + crosshair
    const f = quadCorners(g.s, g.tanX, g.tanY)
    const fl: number[] = []
    for (let i = 0; i < 4; i++) fl.push(...f[i].toArray(), ...f[(i + 1) % 4].toArray())
    const hx = g.s * g.tanX, hy = g.s * g.tanY
    fl.push(-hx * 0.08, 0, -g.s, hx * 0.08, 0, -g.s, 0, -hy * 0.08, -g.s, 0, hy * 0.08, -g.s)
    const focusEdges = new THREE.BufferGeometry(); focusEdges.setAttribute('position', new THREE.Float32BufferAttribute(fl, 3))
    // DoF slab between near and far planes
    const n = quadCorners(g.near, g.tanX, g.tanY), fa = quadCorners(farD, g.tanX, g.tanY)
    const dp: number[] = []
    for (let i = 0; i < 4; i++) {
      const j = (i + 1) % 4
      dp.push(...n[i].toArray(), ...fa[i].toArray(), ...fa[j].toArray(), ...n[i].toArray(), ...fa[j].toArray(), ...n[j].toArray())
    }
    const dof = new THREE.BufferGeometry(); dof.setAttribute('position', new THREE.Float32BufferAttribute(dp, 3))
    return { pyramid, edges, dof, focusEdges }
  }, [g, farD])

  useFrame(({ clock }) => {
    if (matFocus.current) matFocus.current.opacity = acquiring ? 0.16 + 0.1 * (0.5 + 0.5 * Math.sin(clock.elapsedTime * 5)) : 0.14
  })

  return (
    <group>
      <mesh geometry={pyramid} renderOrder={5}>
        <meshBasicMaterial color={FRUSTUM} transparent opacity={0.045} side={THREE.DoubleSide} depthWrite={false} blending={THREE.AdditiveBlending} toneMapped={false} />
      </mesh>
      <lineSegments geometry={edges} renderOrder={6}>
        <lineBasicMaterial color={FRUSTUM} transparent opacity={0.55} toneMapped={false} depthWrite={false} />
      </lineSegments>
      <mesh geometry={dof} renderOrder={5}>
        <meshBasicMaterial color={DOF} transparent opacity={0.09} side={THREE.DoubleSide} depthWrite={false} blending={THREE.AdditiveBlending} toneMapped={false} />
      </mesh>
      <mesh position={[0, 0, -g.s]} renderOrder={6}>
        <planeGeometry args={[2 * g.s * g.tanX, 2 * g.s * g.tanY]} />
        <meshBasicMaterial ref={matFocus} color={FOCUS} transparent opacity={0.14} side={THREE.DoubleSide} depthWrite={false} blending={THREE.AdditiveBlending} toneMapped={false} />
      </mesh>
      <lineSegments geometry={focusEdges} renderOrder={7}>
        <lineBasicMaterial color={FOCUS} transparent opacity={0.95} toneMapped={false} depthWrite={false} />
      </lineSegments>
    </group>
  )
}

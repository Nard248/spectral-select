import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Html, Line, TransformControls } from '@react-three/drei'
import { useFrame, useThree, type ThreeEvent } from '@react-three/fiber'
import * as THREE from 'three'
import { useStudio, activeCamera, activeSource } from '../state/store'
import { lookAtQuaternion } from '../three/pose'
import { wavelengthToHex } from '../lib/color'
import { beamAt } from '../lib/lighting'
import { distance } from '../lib/camera'
import { imagingGeometry } from '../three/hardware/CameraFrustum'
import { useActiveNm } from './useExcitation'
import { useLayerView } from '../state/layerView'
import { totalRange } from '../three/layers/profile'

type V3 = [number, number, number]
const D2R = Math.PI / 180

/** Translate/rotate gizmo for the selected object / source / camera, using a proxy node so every entity (whose pose
 *  is defined by its spec) is edited through the store only. */
export function SelectionGizmo() {
  const sel = useStudio((s) => s.selection)
  const exp = useStudio((s) => s.exp)
  const gizmo = useStudio((s) => s.ui.gizmo)
  const snap = useStudio((s) => s.ui.snap)
  const lookThrough = useStudio((s) => s.ui.lookThrough)
  const edit = useStudio((s) => s.edit)
  const [proxy, setProxy] = useState<THREE.Group | null>(null)
  const dragging = useRef(false)

  const target = useMemo(() => {
    if (!sel) return null
    if (sel.kind === 'object') { const o = exp.objects.find((x) => x.id === sel.id); return o ? { kind: 'object' as const, pos: o.transform.position_mm, rot: o.transform.rotation_deg } : null }
    if (sel.kind === 'source') { const s = exp.sources.find((x) => x.id === sel.id); return s ? { kind: 'hw' as const, pos: s.position_mm, tgt: s.target_mm, roll: s.roll_deg } : null }
    if (sel.kind === 'camera') { const c = exp.cameras.find((x) => x.id === sel.id); return c ? { kind: 'hw' as const, pos: c.position_mm, tgt: c.target_mm, roll: c.roll_deg } : null }
    return null
  }, [sel, exp])

  useLayoutEffect(() => {
    if (!proxy || !target || dragging.current) return
    proxy.position.set(...(target.pos as V3))
    if (target.kind === 'object') proxy.rotation.set(target.rot[0] * D2R, target.rot[1] * D2R, target.rot[2] * D2R, 'XYZ')
    else proxy.quaternion.copy(lookAtQuaternion(target.pos as V3, target.tgt as V3, target.roll))
  }, [proxy, target])

  if (!sel || !target || lookThrough) return null
  const r2 = (v: number) => Math.round(v * 100) / 100

  const onChange = () => {
    if (!proxy) return
    const p = proxy.position
    if (sel.kind === 'object') {
      const e = proxy.rotation
      edit((x) => {
        const o = x.objects.find((q) => q.id === sel.id); if (!o) return
        o.transform.position_mm = [r2(p.x), r2(p.y), r2(p.z)]
        o.transform.rotation_deg = [r2(e.x / D2R), r2(e.y / D2R), r2(e.z / D2R)]
      }, `xf:${sel.id}`)
    } else {
      const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(proxy.quaternion)
      edit((x) => {
        const d = sel.kind === 'source' ? x.sources.find((q) => q.id === sel.id) : x.cameras.find((q) => q.id === sel.id)
        if (!d) return
        const len = distance(d.position_mm, d.target_mm)
        d.position_mm = [r2(p.x), r2(p.y), r2(p.z)]
        if (gizmo === 'rotate') d.target_mm = [r2(p.x + dir.x * len), r2(p.y + dir.y * len), r2(p.z + dir.z * len)]
      }, `xf:${sel.id}`)
    }
  }

  return (
    <group userData={{ hideInPip: true }}>
      <group ref={setProxy} />
      {proxy && <TransformControls object={proxy} mode={gizmo === 'rotate' ? 'rotate' : 'translate'} size={0.75}
        translationSnap={snap ? 25 : null} rotationSnap={snap ? 15 * D2R : null}
        onMouseDown={() => { dragging.current = true }} onMouseUp={() => { dragging.current = false }} onObjectChange={onChange} />}
    </group>
  )
}

/** Drag-to-aim handle at the target of the selected source/camera + aim line. Shift = vertical drag. */
export function AimHandle() {
  const sel = useStudio((s) => s.selection)
  const exp = useStudio((s) => s.exp)
  const lookThrough = useStudio((s) => s.ui.lookThrough)
  const edit = useStudio((s) => s.edit)
  const nm = useActiveNm()
  const { camera, controls } = useThree()
  const drag = useRef<{ plane: THREE.Plane; vertical: boolean; y0: number } | null>(null)
  const ref = useRef<THREE.Mesh>(null)
  const dev = sel && (sel.kind === 'source' ? exp.sources.find((s) => s.id === sel.id) : sel.kind === 'camera' ? exp.cameras.find((c) => c.id === sel.id) : null)
  useFrame(({ clock }) => { if (ref.current) ref.current.scale.setScalar(1 + 0.08 * Math.sin(clock.elapsedTime * 3)) })
  if (!sel || !dev || lookThrough) return null
  const color = sel.kind === 'source' && nm != null ? wavelengthToHex(nm) : '#35d8ff'
  const t = dev.target_mm
  const rr = (v: number) => Math.round(v * 10) / 10

  const down = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation()
    ;(e.target as Element).setPointerCapture?.(e.pointerId)
    const vertical = e.shiftKey
    const n = vertical ? new THREE.Vector3(camera.position.x - t[0], 0, camera.position.z - t[2]).normalize() : new THREE.Vector3(0, 1, 0)
    drag.current = { plane: new THREE.Plane().setFromNormalAndCoplanarPoint(n, new THREE.Vector3(...t)), vertical, y0: t[1] }
    if (controls) (controls as unknown as { enabled: boolean }).enabled = false
  }
  const move = (e: ThreeEvent<PointerEvent>) => {
    const d = drag.current; if (!d) return
    const hit = new THREE.Vector3()
    if (!e.ray.intersectPlane(d.plane, hit)) return
    edit((x) => {
      const o = sel.kind === 'source' ? x.sources.find((q) => q.id === sel.id) : x.cameras.find((q) => q.id === sel.id)
      if (!o) return
      o.target_mm = d.vertical ? [o.target_mm[0], rr(Math.max(-50, hit.y)), o.target_mm[2]] : [rr(hit.x), o.target_mm[1], rr(hit.z)]
    }, `aim:${sel.id}`)
  }
  const up = (e: ThreeEvent<PointerEvent>) => {
    drag.current = null
    ;(e.target as Element).releasePointerCapture?.(e.pointerId)
    if (controls) (controls as unknown as { enabled: boolean }).enabled = true
  }
  return (
    <group userData={{ hideInPip: true }}>
      <Line points={[dev.position_mm as V3, t as V3]} color={color} lineWidth={1} dashed dashSize={6} gapSize={5} transparent opacity={0.55} />
      <mesh ref={ref} position={t as V3} onPointerDown={down} onPointerMove={move} onPointerUp={up}
        onPointerOver={() => { document.body.style.cursor = 'crosshair' }} onPointerOut={() => { document.body.style.cursor = '' }}>
        <sphereGeometry args={[5, 20, 14]} />
        <meshBasicMaterial color={color} transparent opacity={0.9} depthTest={false} />
      </mesh>
      <mesh position={t as V3} rotation-x={Math.PI / 2}><ringGeometry args={[7.5, 8.6, 40]} /><meshBasicMaterial color={color} transparent opacity={0.8} depthTest={false} side={THREE.DoubleSide} /></mesh>
      <Html position={t as V3} zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
        <div className="readout">aim <b>{t.map((v) => Math.round(v)).join(', ')}</b> <span style={{ color: 'var(--faint)' }}>· ⇧ vertical</span></div>
      </Html>
    </group>
  )
}

/** Floating numeric readouts on the selection (beam incidence angle, camera distance, field footprint). */
export function Readouts() {
  const sel = useStudio((s) => s.selection)
  const exp = useStudio((s) => s.exp)
  const lookThrough = useStudio((s) => s.ui.lookThrough)
  const sectionOn = useLayerView((s) => (sel ? s.sections[sel.id]?.on : false)) ?? false
  if (!sel || lookThrough) return null
  const src = activeSource(exp), cam = activeCamera(exp)
  if (sel.kind === 'object') {
    const o = exp.objects.find((x) => x.id === sel.id); if (!o) return null
    const p = o.transform.position_mm
    const blockH = o.shape.kind === 'layered_block' ? totalRange(o.shape).max : 0   // bottom-origin block: readout sits above its top face
    if (o.shape.kind === 'layered_block' && sectionOn) return null                   // the cutaway has its own callouts
    const top: V3 = [p[0], p[1] + blockH + 10, p[2]]
    const b = src ? beamAt(src, [p[0], p[1] + (blockH || 1), p[2]], [0, 1, 0]) : null
    return (
      <group userData={{ hideInPip: true }}><Html position={top} zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
        <div className="readout">
          {b && <div>beam incidence <b>{b.incidenceDeg.toFixed(0)}°</b> · {b.distanceMm.toFixed(0)} mm from lamp</div>}
          {cam && <div>camera distance <b>{distance(p, cam.position_mm).toFixed(0)} mm</b></div>}
        </div>
      </Html></group>
    )
  }
  if (sel.kind === 'camera') {
    const c = exp.cameras.find((x) => x.id === sel.id); if (!c) return null
    const g = imagingGeometry(c)
    const W = 2 * g.s * g.tanX * (g.s - g.f) / g.s * 1, H = 2 * g.s * g.tanY * (g.s - g.f) / g.s * 1
    return <group userData={{ hideInPip: true }}><Html position={[c.position_mm[0], c.position_mm[1] + 50, c.position_mm[2]]} zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
      <div className="readout">field at focus <b>{W.toFixed(0)} × {H.toFixed(0)} mm</b> · WD {c.lens.focus_distance_mm.toFixed(0)} mm</div></Html></group>
  }
  if (sel.kind === 'source') {
    const s = exp.sources.find((x) => x.id === sel.id); if (!s) return null
    return <group userData={{ hideInPip: true }}><Html position={[s.position_mm[0], s.position_mm[1] + 50, s.position_mm[2]]} zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
      <div className="readout">beam ⌀ <b>{s.beam.aperture_diameter_mm} mm</b> · ±{s.beam.divergence_half_angle_deg}° · throw {distance(s.position_mm, s.target_mm).toFixed(0)} mm</div></Html></group>
  }
  return null
}

/** Frame the selection / whole bench with a short animation; and expose pickTable for drag-drop from the library. */
export function ViewBridge({ viewApiRef }: { viewApiRef: { current: import('./viewApi').ViewApi | null } }) {
  const { camera, controls, gl, scene } = useThree()
  useEffect(() => { (window as unknown as { sfThree?: unknown }).sfThree = { scene, camera, gl } }, [scene, camera, gl])
  const goal = useRef<{ pos: THREE.Vector3; tgt: THREE.Vector3 } | null>(null)
  const fly = useRef<{ p0: THREE.Vector3; t0: THREE.Vector3; p1: THREE.Vector3; t1: THREE.Vector3; start: number; dur: number; done: () => void } | null>(null)
  useEffect(() => {
    viewApiRef.current = {
      frame: (center, radius) => {
        const st = useStudio.getState()
        let c = new THREE.Vector3(0, 15, 0), r = 160
        if (center) { c = new THREE.Vector3(...center); r = radius ?? 80 }
        else if (st.selection) {
          const e = st.exp, s = st.selection
          const p = s.kind === 'object' ? e.objects.find((o) => o.id === s.id)?.transform.position_mm
            : s.kind === 'source' ? e.sources.find((o) => o.id === s.id)?.position_mm : s.kind === 'camera' ? e.cameras.find((o) => o.id === s.id)?.position_mm : null
          if (p) { c = new THREE.Vector3(...p); r = s.kind === 'object' ? 60 : 160 }
        } else if (st.exp.objects.length) {
          const ps = st.exp.objects.map((o) => o.transform.position_mm)
          c = new THREE.Vector3(ps.reduce((a, p) => a + p[0], 0) / ps.length, 15, ps.reduce((a, p) => a + p[2], 0) / ps.length)
          r = Math.max(90, ...ps.map((p) => Math.hypot(p[0] - c.x, p[2] - c.z) * 2.1))
        }
        const dir = camera.position.clone().sub((controls as unknown as { target: THREE.Vector3 } | null)?.target ?? new THREE.Vector3()).normalize()
        const fov = ((camera as THREE.PerspectiveCamera).fov * Math.PI) / 180
        goal.current = { tgt: c, pos: c.clone().add(dir.multiplyScalar(r / Math.tan(fov / 2) * 0.62 + r * 0.4)) }
      },
      flyTo: (pos, target, ms = 1800) => new Promise<void>((resolve) => {
        const ctl = controls as unknown as { target: THREE.Vector3 } | null
        fly.current?.done()
        goal.current = null
        fly.current = { p0: camera.position.clone(), t0: (ctl?.target ?? new THREE.Vector3()).clone(), p1: new THREE.Vector3(...pos), t1: new THREE.Vector3(...target), start: performance.now(), dur: Math.max(1, ms), done: resolve }
      }),
      pose: () => {
        const t = (controls as unknown as { target: THREE.Vector3 } | null)?.target ?? new THREE.Vector3()
        return { position: camera.position.toArray(), target: t.toArray() }
      },
      pickTable: (cx, cy) => {
        const rect = gl.domElement.getBoundingClientRect()
        const ndc = new THREE.Vector2(((cx - rect.left) / rect.width) * 2 - 1, -((cy - rect.top) / rect.height) * 2 + 1)
        const rc = new THREE.Raycaster(); rc.setFromCamera(ndc, camera)
        const hit = new THREE.Vector3()
        return rc.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), hit) ? [hit.x, 0, hit.z] : null
      },
    }
    return () => { viewApiRef.current = null }
  }, [camera, controls, gl, viewApiRef])
  // a user drag/zoom interrupts a scripted fly-to
  useEffect(() => {
    const c = controls as unknown as { addEventListener: (t: string, f: () => void) => void; removeEventListener: (t: string, f: () => void) => void } | null
    if (!c?.addEventListener) return
    const stop = () => { const f = fly.current; fly.current = null; f?.done() }
    c.addEventListener('start', stop)
    return () => c.removeEventListener('start', stop)
  }, [controls])
  useFrame((_, dt) => {
    const f = fly.current
    if (f) {
      const ctl = controls as unknown as { target: THREE.Vector3; update: () => void } | null
      const k = Math.min(1, (performance.now() - f.start) / f.dur)
      const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2   // easeInOutCubic
      // slight upward arc on long moves keeps the camera from cutting through the bench
      const arc = Math.sin(Math.PI * e) * 0.06 * f.p0.distanceTo(f.p1)
      camera.position.lerpVectors(f.p0, f.p1, e); camera.position.y += arc
      if (ctl) { ctl.target.lerpVectors(f.t0, f.t1, e); ctl.update() }
      if (k >= 1) { fly.current = null; f.done() }
      return
    }
    const g = goal.current; if (!g) return
    const ctl = controls as unknown as { target: THREE.Vector3; update: () => void } | null
    if (!ctl) return
    const k = 1 - Math.exp(-dt * 7)
    camera.position.lerp(g.pos, k); ctl.target.lerp(g.tgt, k); ctl.update()
    if (camera.position.distanceTo(g.pos) < 0.5) goal.current = null
  })
  return null
}

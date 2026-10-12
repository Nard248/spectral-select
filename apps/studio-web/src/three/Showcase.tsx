import { useMemo } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import presets from '../generated/presets.json'
import type { Experiment, LightSourceSpec, CameraSpec, SceneObject, MaterialSpec } from '../generated/experiment'
import { CameraModel } from './hardware/CameraModel'
import { LightSourceModel } from './hardware/LightSourceModel'
import { SampleMesh } from './SampleMesh'
import { Stage } from './Stage'
import { StudioEnvironment, Effects, type Quality } from './StudioEnvironment'

const exp = (presets.experiments as unknown as Record<string, Experiment>).generic_macro_bench
const mats = (presets as unknown as { materials: MaterialSpec[] }).materials
const mat = (id: string) => mats.find((m) => m.id === id)!

export const SHOWCASE_VIEWS: Record<string, { position: [number, number, number]; target: [number, number, number]; fov?: number }> = {
  overview: { position: [860, 600, 1000], target: [-10, 190, 0], fov: 34 },
  top: { position: [0, 1100, 1], target: [0, 0, 0], fov: 38 },
  camera: { position: [340, 470, 330], target: [0, 440, 0], fov: 30 },
  lamp: { position: [-330, 210, 700], target: [-290, 90, 100], fov: 34 },
  led: { position: [430, 300, 470], target: [150, 150, 170], fov: 32 },
  laser: { position: [330, 270, -400], target: [170, 110, -110], fov: 32 },
  samples: { position: [170, 150, 230], target: [0, 8, 5], fov: 34 },
  lens: { position: [220, 400, -330], target: [0, 360, 0], fov: 30 },
  ledface: { position: [60, 80, 40], target: [170, 205, 190], fov: 32 },
  side: { position: [0, 220, 900], target: [0, 170, 0], fov: 36 },
}

/** logs renderer.info (draw calls / triangles) and mean frame time to the console — enabled with `?stats=1` */
function PerfProbe() {
  const gl = useThree((s) => s.gl)
  const acc = { n: 0, t0: performance.now(), calls: 0, tris: 0 }
  gl.info.autoReset = false
  useFrame(() => {
    const i = gl.info
    acc.calls = i.render.calls; acc.tris = i.render.triangles       // totals of the previous frame (all composer passes)
    i.reset()
    acc.n++
    const now = performance.now()
    if (now - acc.t0 > 3000) {
      console.log(`PERF frames=${acc.n} ms/frame=${((now - acc.t0) / acc.n).toFixed(1)} calls=${acc.calls} tris=${acc.tris} geometries=${i.memory.geometries} textures=${i.memory.textures}`)
      acc.n = 0; acc.t0 = now
    }
  }, -1)
  return null
}

function useScene() {
  return useMemo(() => {
    const base = exp.sources[0]
    const lamp: LightSourceSpec = { ...base, id: 'lamp', kind: 'tunable_lamp', position_mm: [-230, 230, 70], target_mm: [-10, 0, 0],
      beam: { profile: 'gaussian', aperture_diameter_mm: 10, divergence_half_angle_deg: 4 } }
    const led: LightSourceSpec = { ...base, id: 'led', name: 'LED array', kind: 'led_array', position_mm: [170, 210, 190], target_mm: [20, 0, 10],
      beam: { profile: 'flat_top', aperture_diameter_mm: 50, divergence_half_angle_deg: 6 } }
    const laser: LightSourceSpec = { ...base, id: 'laser', name: 'Laser', kind: 'laser', position_mm: [255, 150, -180], target_mm: [40, 4, -40],
      beam: { profile: 'gaussian', aperture_diameter_mm: 4, divergence_half_angle_deg: 0.8 } }
    const camera: CameraSpec = exp.cameras[0]
    const objs: SceneObject[] = [...exp.objects]
    objs.push(
      { id: 'cyl', name: 'cylinder', shape: { kind: 'cylinder', radius_mm: 12, height_mm: 20 }, transform: { position_mm: [90, 10, 60], rotation_deg: [0, 0, 0] }, material_id: 'lichen_green', vessel_material_id: null, visible: true },
      { id: 'gbox', name: 'glass block', shape: { kind: 'box', size_mm: [26, 14, 26] }, transform: { position_mm: [-85, 7, 70], rotation_deg: [0, 30, 0] }, material_id: 'glass', vessel_material_id: null, visible: true },
      { id: 'lich', name: 'lichen', shape: { kind: 'box', size_mm: [30, 8, 22] }, transform: { position_mm: [-10, 4, -80], rotation_deg: [0, -15, 0] }, material_id: 'lichen_orange', vessel_material_id: null, visible: true },
    )
    return { lamp, led, laser, camera, objs }
  }, [])
}

/** Self-contained scene with every 3D model — for QA screenshots and as a reference for integrators.
 *  URL params (showcase.html): `view=overview|top|camera|lamp|led|laser|samples|side`, `acq=1`, `quality=low`. */
export function Showcase({ view = 'overview', acquiring = true, quality = 'high', selected, voidBackground, stats }: { view?: string; acquiring?: boolean; quality?: Quality; selected?: string; voidBackground?: boolean; stats?: boolean }) {
  const { lamp, led, laser, camera, objs } = useScene()
  const v = SHOWCASE_VIEWS[view] ?? SHOWCASE_VIEWS.overview
  const glow: Record<string, string> = { drop: '#6dff7a', bead: '#52ff8a', cuvette: '#ff5040' }
  return (
    <Canvas shadows dpr={[1, 2]} camera={{ position: v.position, fov: v.fov ?? 35, near: 5, far: 14000 }} gl={{ antialias: false, powerPreference: 'high-performance', preserveDrawingBuffer: true }}>
      <StudioEnvironment quality={quality} />
      <Stage background={voidBackground ? { ...exp.background, kind: 'none' } : exp.background} material={mat('black_anodized')} />
      {stats && <PerfProbe />}
      {objs.map((o) => (
        <SampleMesh key={o.id} object={o} material={mat(o.material_id)} vesselMaterial={o.vessel_material_id ? mat(o.vessel_material_id) : undefined}
          glowColor={glow[o.id] ?? null} glowIntensity={0.25} selected={selected === o.id} />
      ))}
      <CameraModel camera={camera} showFrustum acquiring={acquiring} selected={selected === 'camera'} />
      <LightSourceModel source={lamp} activeWavelengthNm={365} showBeam selected={selected === 'lamp'} />
      <LightSourceModel source={led} activeWavelengthNm={488} showBeam />
      <LightSourceModel source={laser} activeWavelengthNm={640} showBeam />
      <OrbitControls makeDefault target={v.target} enableDamping maxDistance={4000} minDistance={30} />
      <Effects quality={quality} />
    </Canvas>
  )
}

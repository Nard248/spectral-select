import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { useStudio, activeCamera } from '../state/store'
import { materialMap } from '../state/presets'
import { Stage, StudioEnvironment, Effects } from '../three'
import { ObjectNodes, SourceNodes, CameraNodes } from './Nodes'
import { SelectionGizmo, AimHandle, Readouts, ViewBridge } from './Gizmos'
import { PipPass } from './RenderLoop'
import { cameraIntrinsics } from '../lib/camera'
import { lookAtQuaternion } from '../three/pose'
import { viewApi } from './viewApi'

export function SceneRoot() {
  const bg = useStudio((s) => s.exp.background)
  const mats = useStudio((s) => s.exp.materials)
  const lookThrough = useStudio((s) => s.ui.lookThrough)
  const quality = useStudio((s) => s.ui.quality)
  const cam = useStudio((s) => activeCamera(s.exp))
  const intr = cameraIntrinsics(cam)
  const material = bg.material_id ? materialMap({ materials: mats })[bg.material_id] : undefined
  return (
    <>
      <StudioEnvironment quality={quality} />
      <Stage background={bg} material={material} />
      <ObjectNodes />
      <SourceNodes />
      <CameraNodes />
      <SelectionGizmo />
      <AimHandle />
      <Readouts />
      <OrbitControls makeDefault enabled={!lookThrough} enableDamping dampingFactor={0.09} minDistance={30} maxDistance={1800}
        maxPolarAngle={Math.PI * 0.497} target={[-10, 200, 0]} rotateSpeed={0.7} />
      <ViewBridge viewApiRef={viewApi} />
      {lookThrough && <PerspectiveCamera makeDefault fov={intr.fovYDeg} near={1} far={8000} position={cam.position_mm}
        quaternion={lookAtQuaternion(cam.position_mm, cam.target_mm, cam.roll_deg)} />}
      <Effects quality={quality} />
      <PipPass />
    </>
  )
}

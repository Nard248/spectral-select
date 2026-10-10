/** Prop contracts for the presentational 3D components (owned by the 3D-models work stream).
 * The app shell passes spec objects straight from the experiment store; components must be
 * pure functions of their props (no store access) so they can be reused in thumbnails. */
import type { CameraSpec, LightSourceSpec, SceneObject, MaterialSpec, Background } from '../generated/experiment'

export interface HardwareCommon {
  selected?: boolean
  hovered?: boolean
}
export interface CameraModelProps extends HardwareCommon {
  camera: CameraSpec
  /** show frustum, focus plane and depth-of-field slab */
  showFrustum?: boolean
  /** true while a capture/preview is running (status LED blinks, filter stack shimmers) */
  acquiring?: boolean
}
export interface LightSourceModelProps extends HardwareCommon {
  source: LightSourceSpec
  /** the excitation currently emitted (drives beam colour and the lamp display), null = off */
  activeWavelengthNm: number | null
  showBeam?: boolean
}
export interface SampleMeshProps extends HardwareCommon {
  object: SceneObject
  material: MaterialSpec
  vesselMaterial?: MaterialSpec
  /** approximate fluorescence glow colour under the active excitation (from the preview shader
   *  / backend), null = no glow */
  glowColor?: string | null
  glowIntensity?: number
}
export interface StageProps {
  background: Background
  material?: MaterialSpec
}

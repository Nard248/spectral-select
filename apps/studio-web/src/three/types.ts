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
  /** layered blocks only: one material per layer (bottom → top), glows per layer, cutaway, annotations, light-field texture */
  layerMaterials?: MaterialSpec[]
  layerGlows?: (LayerGlow | null)[]
  section?: BlockSection | null
  /** draw layer names / thickness callouts on the section face */
  sectionLabels?: boolean
  /** colour-mapped light field painted on the section face; needs `section` (see `scene/lightfield.ts`) */
  lightField?: import('three').Texture | null
}
export interface StageProps {
  background: Background
  material?: MaterialSpec
}

/** ---- v1.1 layered block (optional props, added by the weblayers work stream) ---- */
/** Cutaway of a layered block: the half with u > position (keep 'low') or u < position (keep 'high') is removed. */
export interface BlockSection {
  axis: 'x' | 'z'
  /** local coordinate of the cut plane along `axis` (mm, block-local frame) */
  position_mm: number
  keep: 'low' | 'high'
}
/** Fluorescence glow of one layer under the active excitation (same model as `SampleMeshProps.glowColor`). */
export interface LayerGlow { color: string | null; intensity: number }

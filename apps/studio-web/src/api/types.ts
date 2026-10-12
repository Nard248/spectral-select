/**
 * HTTP/WebSocket contract of the SpectralForge Studio engine (FastAPI, `src/spectraforge/studio/server.py`).
 * Base URL: same origin under `/api` (Vite dev server proxies `/api` → http://127.0.0.1:8765).
 *
 *   GET  /api/health                          → Health
 *   GET  /api/presets                         → PresetsBundle   (same content as src/generated/presets.json)
 *   POST /api/validate      body Experiment   → ValidationResult
 *   POST /api/preview       body PreviewRequest → CaptureMeta   (synchronous, small; one excitation, scaled res)
 *   POST /api/captures      body CaptureRequest → { job_id }
 *   WS   /api/jobs/{job_id}                   → stream of JobEvent (closes after done/error/cancelled)
 *   GET  /api/jobs/{job_id}                   → JobStatus        (polling fallback)
 *   POST /api/jobs/{job_id}/cancel            → JobStatus
 *   GET  /api/captures                        → CaptureMeta[]    (this server session, newest first)
 *   GET  /api/captures/{id}                   → CaptureMeta
 *   GET  /api/captures/{id}/band?ex=&band=&kind=adu|radiance|electrons
 *                                             → application/octet-stream float32 LE, H*W row-major
 *   GET  /api/captures/{id}/cube?ex=&kind=    → float32 LE, H*W*B (band fastest)
 *   GET  /api/captures/{id}/image.png?mode=true_color|false_color|band&ex=&band=&gain=
 *                                             → PNG (server-side colour rendering)
 *   GET  /api/captures/{id}/pixel?x=&y=       → PixelExplain
 *   GET  /api/captures/{id}/gt/{name}         → float32 LE, H*W (name ∈ CaptureMeta.ground_truth_layers)
 *   GET  /api/captures/{id}/download?format=spectradata_pkl|npz|tiff_zip|ground_truth_zip|experiment_json
 *                                             → file attachment
 * Binary responses carry headers X-Width, X-Height, X-Bands (cube only).
 */
import type { Experiment } from '../generated/experiment'

export interface Health { ok: boolean; engine_version: string; schema_version: string }

export interface PresetsBundle {
  schema_version: string
  extinction_reference_M_cm: number
  fluorophores: import('../generated/experiment').FluorophoreSpec[]
  materials: import('../generated/experiment').MaterialSpec[]
  experiments: Record<string, Experiment>
}

export interface ValidationIssue { loc: (string | number)[]; msg: string }
export interface PhysicsWarning { code: string; severity: 'info' | 'warn'; message: string; target_id?: string }
export interface ValidationResult { ok: boolean; errors: ValidationIssue[]; warnings: PhysicsWarning[] }

export interface PreviewRequest {
  experiment: Experiment
  excitation_nm?: number | null   // default: first enabled step
  scale?: number | null           // default: experiment.acquisition.preview_scale
}
export interface CaptureRequest {
  experiment: Experiment
  mode?: 'sample' | 'dark' | 'white_reference'
}

export type JobEvent =
  | { type: 'progress'; fraction: number; message: string }
  | { type: 'done'; capture_id: string }
  | { type: 'error'; message: string }
  | { type: 'cancelled' }

export interface JobStatus {
  job_id: string
  state: 'queued' | 'running' | 'done' | 'error' | 'cancelled'
  fraction: number
  message: string
  capture_id?: string | null
  error?: string | null
}

export interface BandStats { min: number; max: number; p01: number; p99: number; mean: number }

export interface CaptureMeta {
  id: string
  created: string                       // ISO time
  name: string                          // experiment name
  mode: 'sample' | 'dark' | 'white_reference' | 'preview'
  width: number
  height: number
  excitations_nm: number[]
  emission_nm: Record<string, number[]> // key = String(excitation)
  exposure_ms: Record<string, number>
  power_mW: Record<string, number>
  stats: Record<string, BandStats[]>    // per excitation, per band (ADU)
  saturated_fraction: Record<string, number>
  ground_truth_layers: string[]
  warnings: string[]
  timings_s: Record<string, number>
  camera_id: string
  source_id: string
}

export interface PixelExplain {
  x: number
  y: number
  emission_nm: Record<string, number[]>
  adu: Record<string, number[]>                       // per excitation, per band
  radiance: Record<string, number[]>
  /** noise-free per-component radiance, per excitation → component name → per band.
   *  Component names: `dye:<fluorophore>`, `secondary:<fluorophore>`, `elastic`, `glint`,
   *  `raman`, `leak_source` (excitation-filter leakage), `background:<...>`, `room_light`. */
  components: Record<string, Record<string, number[]>>
  ground_truth: Record<string, number | string | null>   // material, object, depth_mm, angles, …
}

// =============================================================================================
// v1.1 — analysis tools, 3D layered-block light fields, parameter sweeps
// =============================================================================================
//   POST /api/captures/{id}/eem        body EemRequest      → EemResult        (pixel / disk / ROI EEM)
//   POST /api/captures/{id}/roi        body RoiRequest      → RoiStats
//   POST /api/captures/{id}/line       body LineRequest     → LineProfile
//   POST /api/captures/{id}/response   body ResponseRequest → ResponseCurve    (metric vs a GT map)
//   GET  /api/captures/{id}/volumes                         → VolumeInfo[]
//   GET  /api/captures/{id}/volume?object=&kind=&ex=&channel=
//        → float32 LE, (nx, nz, n_layers·sublayers) C-order, headers X-Dims "nx,nz,nd"
//   POST /api/sweeps                   body SweepRequest    → { sweep_id }
//   WS   /api/sweeps/{sweep_id}                             → stream of SweepEvent
//   GET  /api/sweeps/{sweep_id}                             → SweepStatus
//   POST /api/sweeps/{sweep_id}/cancel                      → SweepStatus
// CaptureMeta gains optional `volumes?: VolumeInfo[]` and `blocks?: BlockInfo[]` (below).

/** How pixel values are reported by analysis endpoints. */
export interface NormSpec {
  /** irradiance_normalized: (ADU − offset) / (exposure_s × E_ex) with E_ex the ground-truth surface
   *  irradiance `irradiance@<ex>` (mW/mm²) at each pixel — removes the lamp's spatial gradient
   *  (recommended for signal-vs-structure studies on obliquely lit samples); NaN where
   *  E_ex < 1e-3 · max(E_ex) (beam tails / shadow, where the ratio is numerically meaningless) */
  mode: 'adu' | 'radiance' | 'electrons' | 'dark_subtracted' | 'exposure_power' | 'white_normalized'
      | 'irradiance_normalized'
  /** required for dark_subtracted (and used by white_normalized if given) */
  dark_capture_id?: string | null
  /** required for white_normalized: (sample − dark) / (white − dark) */
  white_capture_id?: string | null
}

/** Region of interest in capture pixel coordinates (x = column, y = row, origin top-left). */
export type RoiSpec =
  | { shape: 'point'; x: number; y: number }
  | { shape: 'disk'; x: number; y: number; radius_px: number }
  | { shape: 'rect'; x0: number; y0: number; x1: number; y1: number }
  | { shape: 'ellipse'; cx: number; cy: number; rx: number; ry: number }
  | { shape: 'polygon'; points: [number, number][] }

export interface EemRequest { roi: RoiSpec; norm?: NormSpec | null; stat?: 'mean' | 'median' }
export interface EemResult {
  roi: RoiSpec
  n_pixels: number
  norm: NormSpec
  unit: string
  excitations_nm: number[]
  /** per excitation: band centres (they differ per excitation under the Nuance band policy) */
  emission_nm: Record<string, number[]>
  /** per excitation: per-band statistic over the ROI */
  value: Record<string, number[]>
  std: Record<string, number[]>
  /** EEM resampled on a common emission grid (union span, 2 nm) for heatmaps; NaN where a band is absent */
  grid_emission_nm: number[]
  grid_value: number[][]           // [n_ex][n_grid]
}

export interface RoiStats extends EemResult {
  /** ground-truth map statistics inside the ROI (numeric layers only): mean/std/min/max */
  ground_truth: Record<string, { mean: number; std: number; min: number; max: number }>
  /** optional averaged component decomposition (subsampled ≤ 64 pixels), like PixelExplain.components */
  components?: Record<string, Record<string, number[]>> | null
}
export interface RoiRequest extends EemRequest { components?: boolean }

export interface LineRequest {
  p0: [number, number]; p1: [number, number]
  width_px?: number                 // averaging width perpendicular to the line (default 1)
  norm?: NormSpec | null
  excitations_nm?: number[] | null  // default all
  ground_truth?: string[] | null    // GT layers to sample along the line
}
export interface LineProfile {
  distance_px: number[]
  distance_mm: number[] | null      // on the first surface, if depth is known
  /** per excitation → [n_points][n_bands] */
  value: Record<string, number[][]>
  emission_nm: Record<string, number[]>
  ground_truth: Record<string, number[]>
  unit: string
}

/** A scalar per pixel derived from the cube. */
export type MetricSpec =
  | { kind: 'band'; ex: number; band_nm: number }
  | { kind: 'band_ratio'; a: { ex: number; band_nm: number }; b: { ex: number; band_nm: number } }
  | { kind: 'band_sum'; ex: number; from_nm?: number; to_nm?: number }
  | { kind: 'peak_wavelength'; ex: number }
export interface ResponseRequest {
  x_ground_truth: string            // e.g. 'layer_thickness:cover'
  metric: MetricSpec
  roi?: RoiSpec | null
  bins?: number                     // default 24
  norm?: NormSpec | null
}
export interface ResponseCurve {
  x_ground_truth: string
  metric: MetricSpec
  bin_centres: number[]
  mean: number[]; std: number[]; count: number[]
  /** raw scatter subsample (≤ 4000 points) */
  scatter_x: number[]; scatter_y: number[]
  unit: string
}

/** A layered block present in the capture (for analysis labels and the light-field viewer). */
export interface BlockInfo {
  object_id: string
  layer_names: string[]
  footprint_mm: [number, number]
}
export interface VolumeInfo {
  object_id: string
  kind: 'fluence' | 'absorbed' | 'source'
  excitations_nm: number[]
  channels: string[]                // for 'source' (e.g. 'tissue|NADH'); [] otherwise
  dims: [number, number, number]    // nx, nz, n_layers·sublayers
  x_edges_mm: number[]; z_edges_mm: number[]
  layer_names: string[]; sublayers: number
  unit: string
}

export interface SweepRequest {
  experiment: Experiment
  /** JSON pointer into the experiment, e.g. "/objects/0/shape/layers/2/thickness/thickness_mm" */
  parameter: { path: string; values: (number | string | boolean)[]; label?: string }
  mode?: 'preview' | 'sample'       // default 'preview'
}
export type SweepEvent =
  | { type: 'progress'; fraction: number; message: string; index: number }
  | { type: 'point'; index: number; value: number | string | boolean; capture_id: string }
  | { type: 'done'; sweep_id: string }
  | { type: 'error'; message: string }
  | { type: 'cancelled' }
export interface SweepStatus {
  sweep_id: string
  state: 'queued' | 'running' | 'done' | 'error' | 'cancelled'
  fraction: number
  parameter: SweepRequest['parameter']
  capture_ids: (string | null)[]
  error?: string | null
}

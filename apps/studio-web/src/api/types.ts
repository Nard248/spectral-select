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

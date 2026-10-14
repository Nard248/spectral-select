/**
 * v1.2 Research workspace: HTTP/WS contract of the study, sensitivity, methods, provenance, inverse and material-optics
 * endpoints. Mirrors, field for field, the Pydantic models in `src/spectraforge/studio/study_schema.py` and the route
 * modules `routes_study.py`, `routes_inverse.py`, `routes_materials.py` (documented in docs/studio/STUDY_API.md,
 * INVERSE_API.md, SIM2REAL.md).
 *
 *   POST /api/studies[?keep_captures=1]    StudySpec            → { study_id, n_runs, n_points }
 *   GET  /api/studies                                           → StudyStatus[]
 *   GET  /api/studies/{id}                                      → StudyStatus (+ manifest once terminal)
 *   WS   /api/studies/{id}/events                               → StudyEvent stream
 *   POST /api/studies/{id}/cancel                               → StudyStatus
 *   GET  /api/studies/{id}/runs?offset=&limit=                  → StudyRuns (409 until the first results exist)
 *   GET  /api/studies/{id}/sensitivity?boot=                    → SensitivityResult (saltelli / morris designs only)
 *   POST /api/methods_text                                      → { style, text }
 *   GET  /api/provenance                                        → Provenance
 *   POST /api/material_optics                                   → MaterialOptics
 *   GET  /api/calibrated_fluorophores                           → CalibratedFluorophore[]
 *   POST /api/unmix, GET /api/unmix/{id}/map/{name}             → UnmixResult, float32 map
 *   POST /api/surrogates (job), POST /api/fit, POST /api/fisher (sync | job), POST /api/design, GET /api/inverse/results/{id}
 */
import type { Experiment } from '../generated/experiment'
import type { NormSpec, RoiSpec } from './types'

// ------------------------------------------------------------------------------------------------ study spec
export type StudyScale = 'linear' | 'log'
export type ParameterKind = 'continuous' | 'integer' | 'choice'
export interface ParameterSpec {
  pointer: string
  label?: string | null
  kind: ParameterKind
  low?: number | null
  high?: number | null
  scale: StudyScale
  values?: (string | number | boolean)[] | null
  also_set: string[]
}

export type DesignKind = 'grid' | 'random' | 'lhs' | 'sobol' | 'saltelli' | 'morris'
export interface DesignSpec { kind: DesignKind; n: number; levels: number; seed: number }

export type MetricKind = 'roi_band' | 'band_ratio' | 'band_sum' | 'peak_wavelength' | 'eem_integral' | 'gt_stat' | 'contrast' | 'cnr'
export type Stat = 'mean' | 'median' | 'std' | 'min' | 'max'
export interface BandRef { ex: number; band_nm: number }
export interface MetricSpec {
  name: string
  kind: MetricKind
  roi?: RoiSpec | null
  roi_b?: RoiSpec | null
  roi_units: 'fraction' | 'pixels'
  norm: NormSpec
  stat: Stat
  ex?: number | null
  band_nm?: number | null
  a?: BandRef | null
  b?: BandRef | null
  from_nm?: number | null
  to_nm?: number | null
  gt?: string | null
  inner?: Record<string, unknown> | null
  contrast: 'michelson' | 'weber'
}

export type StudyMode = 'preview' | 'fast' | 'full'
export interface StudySpec {
  name: string
  base_experiment: Experiment | string
  parameters: ParameterSpec[]
  design: DesignSpec
  replicates: number
  mode: StudyMode
  excitations?: number[] | null
  metrics: MetricSpec[]
  max_runs: number
}

// ------------------------------------------------------------------------------------------------ study run state
export type StudyState = 'queued' | 'running' | 'done' | 'error' | 'cancelled'
export interface StudyManifest {
  spec?: StudySpec
  spec_hash?: string
  provenance?: Provenance & { experiment_hash?: string }
  n_runs?: number
  n_completed?: number
  n_failed?: number
  timings_s?: Record<string, number>
  [k: string]: unknown
}
export interface StudyStatus {
  study_id: string
  name: string
  state: StudyState
  fraction: number
  message: string
  n_runs: number
  error: string | null
  manifest?: StudyManifest
}
export type StudyEvent =
  | { type: 'progress'; fraction: number; message: string }
  | { type: 'done'; study_id: string; n_failed: number }
  | { type: 'error'; message: string }
  | { type: 'cancelled' }
export interface StudyStart { study_id: string; n_runs: number; n_points: number }

/** One row per (design point, replicate). `p:*` parameter value, `u:*` unit-cube coordinate, `m:*` metric. NaN arrives as null. */
export interface StudyRunRow {
  run_id: string
  point: number
  replicate: number
  seed: number
  exp_hash?: string
  block?: string | null
  row?: number | null
  traj?: number | null
  step?: number | null
  step_param?: string | null
  direction?: number | null
  status: 'ok' | 'error' | string
  error?: string | null
  t_capture_s?: number | null
  t_metrics_s?: number | null
  warnings?: string | null
  [col: string]: string | number | boolean | null | undefined
}
export interface StudyRuns { total: number; columns: string[]; rows: StudyRunRow[] }

// ------------------------------------------------------------------------------------------------ sensitivity
export interface SobolIndex { S1: number | null; S1_ci: [number, number] | null; ST: number | null; ST_ci: [number, number] | null }
export interface MorrisIndex { mu: number | null; mu_star: number | null; mu_star_ci: [number, number] | null; sigma: number | null; n_effects: number }
export interface SobolMetric {
  method: 'sobol'
  n_rows: number
  variance: number | null
  sum_S1: number | null
  noise: { replicates: number; within_var: number | null; noise_floor: number | null }
  indices: Record<string, SobolIndex>
}
export interface MorrisMetric {
  method: 'morris'
  n_rows?: number
  variance?: number | null
  noise?: { replicates: number; within_var: number | null; noise_floor: number | null }
  indices: Record<string, MorrisIndex>
}
export interface SensitivityResult {
  design: 'saltelli' | 'morris'
  parameters: string[]
  metrics: Record<string, SobolMetric | MorrisMetric>
}

// ------------------------------------------------------------------------------------------------ methods / provenance
export type MethodsStyle = 'markdown' | 'latex'
export interface MethodsRequest { experiment: Experiment; study?: StudySpec | null; style: MethodsStyle }
export interface MethodsText { style: MethodsStyle; text: string }
export interface Provenance {
  package: string
  package_version: string
  engine_version: string
  git: { commit: string | null; branch: string | null; dirty: boolean | null }
  python: string
  numpy: string
  scipy: string
  numba?: { version: string; threading_layer: string | null; num_threads: number | null } | string | null
  platform: string
  machine: string
  cpu_count: number
  timestamp_utc: string
  [k: string]: unknown
}

// ------------------------------------------------------------------------------------------------ material optics (engine-backed)
export interface MaterialOpticsRequest { material: unknown; fluorophores?: unknown[]; nm_min?: number; nm_max?: number; step?: number }
export interface MaterialOptics {
  nm: number[]
  mua_per_mm: number[]
  mus_prime_per_mm: number[]
  g: number
  n: number
  /** `dye:<name>`, `absorber`, `hbo2`, `hb`, `water`, `lipid`, `melanosome`, `bilirubin`, `beta_carotene` → μa (1/mm) */
  components: Record<string, number[]>
}

// ------------------------------------------------------------------------------------------------ calibrated fluorophores (W6)
export interface CalibratedProvenance {
  dataset?: string
  roi?: string
  method?: string
  instrument?: string
  r2?: number | null
  sam_mean_deg?: number | null
  rms_rel?: number | null
  n_obs?: number | null
  ci95?: Record<string, number | null>
  notes?: string
  [k: string]: unknown
}
export interface CalibratedFluorophore {
  spec: import('../generated/experiment').FluorophoreSpec
  provenance: CalibratedProvenance
}

// ------------------------------------------------------------------------------------------------ inverse problems
export interface InverseParameter { name?: string; path?: string; paths?: string[]; mode?: 'absolute' | 'shift'; lo?: number; hi?: number }
export interface ChannelTable { ex_nm: number[]; em_nm: number[] }
export type JobResultId = string

export type UnmixMethod = 'nnls' | 'fcls' | 'ls'
export type EndmemberSource = 'forward_model' | 'rois' | 'nmf' | 'vca'
export interface UnmixRequest {
  capture_id: string
  method?: UnmixMethod
  endmembers?: { source: EndmemberSource; groups?: unknown; rois?: Record<string, RoiSpec>; n_components?: number; seed?: number }
  weighting?: 'none' | 'poisson'
  signal?: 'adu_net' | 'electrons' | 'radiance'
  excitations?: number[]
  compare_gt?: { n_pixels: number; seed: number }
  include_maps?: boolean
}
export interface GtComparison { pearson_r: number | null; slope: number | null; rmse: number | null; rel_rmse: number | null }
export interface UnmixResult {
  names: string[]
  method: string
  endmembers: { names: string[]; source: string; signal: string; channels: ChannelTable; matrix: number[][]; meta: Record<string, unknown> }
  meta: Record<string, unknown>
  shape: [number, number]
  mean_residual_fraction: number | null
  mean_abundance: Record<string, number>
  gt_comparison?: Record<string, GtComparison>
  unmix_id: string
}

export interface SurrogateRequest {
  experiment: Experiment
  parameters: InverseParameter[]
  mode?: 'grid' | 'sobol'
  grid: Record<string, number[]>
  n_points?: number
  roi?: Record<string, unknown>
  scale?: number
  excitations?: number[]
  overrides?: Record<string, unknown>
  smooth?: string
  cache?: boolean
}
export interface SurrogateSummary {
  surrogate_id: string
  names: string[]
  n_channels: number
  n_points: number
  bounds: number[][]
  mode: string
  loo?: { loo_rel_rms_mean: number; loo_rel_rms_max: number; n: number } | null
  noise?: unknown
  meta?: Record<string, unknown>
}

export interface FitRequest {
  surrogate_id: string
  observation: { capture_id: string; roi?: Record<string, unknown>; kind?: string } | { values_e: number[] }
  x0?: number[]
  bounds?: number[][]
  profile?: boolean
  level?: number
  map?: { tile: number; gt_name?: string; min_signal_e?: number }
}
export interface FitCi { lo: number | null; hi: number | null; open_lo?: boolean; open_hi?: boolean; level: number }
export interface FitResult {
  names: string[]
  theta: number[]
  se: (number | null)[]
  cov: number[][]
  chi2: number
  dof: number
  reduced_chi2: number
  success?: boolean
  at_bound: boolean[]
  ci: Record<string, FitCi>
  profile: Record<string, { values: number[]; delta_chi2: number[] }> | null
  n_roi_pixels?: number | null
}
export interface MapStats { bias?: number | null; rmse?: number | null; pearson_r?: number | null; coverage_1p96se?: number | null; n_tiles?: number | null; [k: string]: number | null | undefined }
export interface FitMapResult {
  names: string[]
  tile: number
  shape: [number, number]
  stats: MapStats | Record<string, MapStats>
  /** [rows][cols][P] with null = not fitted */
  theta: (number | null)[][][] | (number | null)[][]
  se: unknown
  chi2_red: (number | null)[][]
  gt: (number | null)[][] | null
  map_id?: string
}

export interface FisherRequest {
  surrogate_id?: string
  theta?: number[]
  experiment?: Experiment
  parameters?: InverseParameter[]
  bands?: 'all' | { from_nm: number; to_nm: number } | number[]
  excitations?: number[]
  roi?: Record<string, unknown>
  step?: number
  theta0?: number[]
  scale?: number
  overrides?: Record<string, unknown>
  scheme?: 'central' | 'forward'
}
export interface FisherResult {
  fisher_id: string
  names: string[]
  theta: number[]
  fisher: number[][]
  crlb_covariance: (number | null)[][]
  crlb_sd: (number | null)[]
  crlb_rel_sd: (number | null)[]
  identifiable: boolean
  n_channels: number
  channels: ChannelTable
  mean_e: number[]
  /** (C, P) derivative of the expected electrons with respect to each parameter */
  jac_e: number[][]
  var_e: number[]
  /** (C, P) diagonal information J_cp² / var_c */
  info_per_channel: number[][]
  meta: Record<string, unknown>
}

export type DesignCriterion = 'D' | 'A' | `target:${string}`
export interface DesignRequest {
  fisher_id?: string
  surrogate_id?: string
  theta?: number[]
  k: number
  criterion?: DesignCriterion
  budget?: 'channels' | 'exposure_time'
  total_time_s?: number
  bands?: FisherRequest['bands']
  excitations?: number[]
}
export interface ExposureAllocation {
  total_time_s: number
  channels: ChannelTable
  t_opt_s: number[]
  t_reference_s: number[]
  t_equal_s: number[]
  crlb_sd_opt: number[]
  crlb_sd_reference: number[]
  crlb_sd_equal: number[]
  t_max_s: number[]
  names: string[]
}
export interface DesignResult {
  criterion: string
  k: number
  selected: number[]
  channels: ChannelTable
  labels: string[]
  objective: number[]
  names: string[]
  /** (k, P): CRLB sd after 1..k greedily chosen channels */
  crlb_sd_vs_k: (number | null)[][]
  fisher: number[][]
  exhaustive: { set: number[]; objective: number; greedy_objective: number; refined_objective: number; greedy_same_set: boolean; same_set: boolean; greedy_gap: number; refined_gap: number; n_combinations: number } | null
  exposure: ExposureAllocation | null
  all_channels_sd: (number | null)[]
}

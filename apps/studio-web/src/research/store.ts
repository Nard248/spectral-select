/** Research workspace state: the study being built, the running / finished study with its runs and sensitivity indices,
 *  the Methods text and the inverse-problem configuration and results. Side effects live in `actions.ts` / `inverseActions.ts`. */
import { create } from 'zustand'
import type {
  CalibratedFluorophore, DesignCriterion, DesignResult, DesignSpec, FisherResult, FitMapResult, FitResult, MetricSpec, MethodsStyle, ParameterSpec,
  Provenance, SensitivityResult, StudyManifest, StudyMode, StudyRunRow, StudySpec, StudyState, StudyStatus, SurrogateSummary, UnmixMethod, UnmixResult, EndmemberSource,
} from '../api/researchTypes'
import type { RoiSpec } from '../api/types'

export type ResearchTab = 'study' | 'inverse' | 'methods'
export type InverseTab = 'unmix' | 'info' | 'design' | 'fit'

/** A parameter row in the builder: the spec plus what the UI needs to render it. */
export interface ParamDraft extends ParameterSpec { unit: string; group: string; current: number | null }

export interface StudyRun {
  id: string
  spec: StudySpec
  state: StudyState
  fraction: number
  message: string
  error: string | null
  nRuns: number
  nPoints: number
  startedAt: number
  manifest: StudyManifest | null
}

export interface InverseParam { path: string; paths: string[]; name: string; label: string; unit: string; value: number; lo: number; hi: number; mode: 'absolute' | 'shift' }
export interface Busy<T> { state: 'idle' | 'running' | 'done' | 'error'; fraction: number; message: string; error: string | null; result: T | null }
const idle = <T,>(): Busy<T> => ({ state: 'idle', fraction: 0, message: '', error: null, result: null })

export interface InverseState {
  tab: InverseTab
  params: InverseParam[]
  /** ROI in image fractions (normalized) shared by the Fisher job, surrogate and ROI fit */
  roi: RoiSpec
  roiId: string | null
  captureId: string | null
  excitations: number[] | null
  bandFrom: number | null
  bandTo: number | null
  // unmix
  unmixMethod: UnmixMethod
  endmemberSource: EndmemberSource
  nComponents: number
  weighting: 'none' | 'poisson'
  compareGt: boolean
  unmix: Busy<UnmixResult>
  unmixMaps: Record<string, { w: number; h: number; data: Float32Array }>
  // information / design
  gridPoints: number
  fisher: Busy<FisherResult>
  k: number
  criterion: DesignCriterion
  budget: 'channels' | 'exposure_time'
  totalTimeS: number
  design: Busy<DesignResult>
  targetParam: number
  // surrogate / fit
  surrogate: Busy<SurrogateSummary>
  fit: Busy<FitResult>
  fitMap: Busy<FitMapResult>
  tile: number
}

interface ResearchState {
  tab: ResearchTab
  // ---- study builder
  name: string
  params: ParamDraft[]
  design: DesignSpec
  replicates: number
  mode: StudyMode
  excitations: number[] | null
  metrics: MetricSpec[]
  keepCaptures: boolean
  touched: boolean
  // ---- run + results
  run: StudyRun | null
  columns: string[]
  rows: StudyRunRow[]
  total: number
  sens: SensitivityResult | null
  sensError: string | null
  sensLoading: boolean
  metricSel: string | null
  xParam: string | null
  selectedRun: string | null
  brush: Record<string, [number, number]>
  history: StudyStatus[]
  observed: Partial<Record<StudyMode, number>>
  // ---- methods + provenance
  methodsStyle: MethodsStyle
  methodsText: string
  methodsFor: 'experiment' | 'study'
  methodsBusy: boolean
  methodsError: string | null
  provenance: Provenance | null
  // ---- calibrated fluorophores
  calibrated: CalibratedFluorophore[] | null
  // ---- inverse
  inv: InverseState
  set: (p: Partial<ResearchState>) => void
  setInv: (p: Partial<InverseState>) => void
}

export const DEFAULT_DESIGN: DesignSpec = { kind: 'saltelli', n: 8, levels: 4, seed: 0 }

export const useResearch = create<ResearchState>()((set) => ({
  tab: 'study',
  name: 'study',
  params: [],
  design: DEFAULT_DESIGN,
  replicates: 1,
  mode: 'fast',
  excitations: null,
  metrics: [],
  keepCaptures: false,
  touched: false,
  run: null, columns: [], rows: [], total: 0,
  sens: null, sensError: null, sensLoading: false,
  metricSel: null, xParam: null, selectedRun: null, brush: {},
  history: [], observed: {},
  methodsStyle: 'markdown', methodsText: '', methodsFor: 'experiment', methodsBusy: false, methodsError: null, provenance: null,
  calibrated: null,
  inv: {
    tab: 'info', params: [], roi: { shape: 'rect', x0: 0.3, y0: 0.3, x1: 0.7, y1: 0.7 }, roiId: null, captureId: null, excitations: null, bandFrom: null, bandTo: null,
    unmixMethod: 'nnls', endmemberSource: 'forward_model', nComponents: 3, weighting: 'none', compareGt: true, unmix: idle(), unmixMaps: {},
    gridPoints: 9, fisher: idle(), k: 8, criterion: 'D', budget: 'channels', totalTimeS: 4, design: idle(), targetParam: 0,
    surrogate: idle(), fit: idle(), fitMap: idle(), tile: 8,
  },
  set: (p) => set(p),
  setInv: (p) => set((s) => ({ inv: { ...s.inv, ...p } })),
}))

export { idle }
if (import.meta.env.DEV && typeof window !== 'undefined') (window as unknown as { research: typeof useResearch }).research = useResearch

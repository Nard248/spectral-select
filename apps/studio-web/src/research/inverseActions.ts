/** Side effects of the inverse-problem panel: unmixing, Fisher information / CRLB, optimal channel design, surrogate + fits.
 *  Anything that runs the simulator is a job on the shared worker; its `done` event carries the id of the stored result. */
import type { InverseParameter, FisherResult, DesignResult, FitResult, FitMapResult, SurrogateSummary, UnmixResult } from '../api/researchTypes'
import { getEngine } from '../api/client'
import { useStudio } from '../state/store'
import { toast } from '../ui/toast'
import type { ParamOption } from '../sweep/paths'
import { loadResearchApi, isJob } from './api'
import { defaultRange, suggestLinks } from './logic'
import { useResearch, type Busy, type InverseParam, type InverseState } from './store'
import { analysisRois, referenceMeta, rememberRoi } from './actions'
import type { RoiSpec } from '../api/types'

type Key = 'unmix' | 'fisher' | 'design' | 'surrogate' | 'fit' | 'fitMap'
const setBusy = <T,>(key: Key, b: Partial<Busy<T>>) => useResearch.setState((s) => ({ inv: { ...s.inv, [key]: { ...(s.inv[key] as Busy<T>), ...b } } }))
const errText = (e: unknown) => (e as Error).message.replace(/^(GET|POST) [^\s]+ → \d+ /, '').slice(0, 400)

let jobCancel: (() => void) | null = null
export function cancelInverseJob(jobId: string | null) {
  const eng = getEngine()
  if (jobId && eng) void eng.cancelJob(jobId).catch(() => undefined)
  jobCancel?.()
}
let activeJob: string | null = null
export const currentJob = () => activeJob

/** Follow a job to completion; resolves with the stored result id (the `capture_id` field of the done event). */
export async function followJob(jobId: string, key: Key): Promise<string> {
  const api = await loadResearchApi()
  activeJob = jobId
  return new Promise<string>((resolve, reject) => {
    const un = api.watchJob(jobId, (e) => {
      if (e.type === 'progress') setBusy(key, { fraction: e.fraction, message: e.message })
      else { un(); jobCancel = null; activeJob = null; if (e.type === 'done') resolve(e.capture_id); else reject(new Error(e.type === 'error' ? e.message : 'cancelled')) }
    })
    jobCancel = () => { un(); reject(new Error('cancelled')) }
  })
}

// ---------------------------------------------------------------------------------------------- parameters / ROI
export function invParamFromOption(o: ParamOption, all: string[]): InverseParam {
  const r = defaultRange(o.value, o.unit, o.label)
  const links = suggestLinks(o.path, all)
  const label = o.label.split(' › ').slice(-2).join(' › ')
  return { path: o.path, paths: [o.path, ...links], name: label.replace(/[^\w.\- ]/g, '').replace(/\s+/g, '_'), label, unit: o.unit, value: o.value, lo: r.low, hi: r.high, mode: 'absolute' }
}
export function addInvParam(o: ParamOption, all: string[]) {
  const st = useResearch.getState()
  if (st.inv.params.some((p) => p.path === o.path)) return
  if (st.inv.params.length >= 8) { toast('At most 8 parameters per inverse problem', 'error'); return }
  let p = invParamFromOption(o, all)
  const taken = new Set(st.inv.params.map((q) => q.name)); let i = 2; const base = p.name
  while (taken.has(p.name)) p = { ...p, name: `${base}_${i++}` }
  st.setInv({ params: [...st.inv.params, p] })
}
export function patchInvParam(i: number, patch: Partial<InverseParam>) {
  const st = useResearch.getState()
  st.setInv({ params: st.inv.params.map((p, j) => (j === i ? { ...p, ...patch } : p)) })
}
export const removeInvParam = (i: number) => { const st = useResearch.getState(); st.setInv({ params: st.inv.params.filter((_, j) => j !== i) }) }

export function paramsSpec(inv: InverseState = useResearch.getState().inv): InverseParameter[] {
  return inv.params.map((p) => ({ name: p.name, paths: p.paths, mode: p.mode, lo: p.lo, hi: p.hi }))
}
export const roiDict = (roi: RoiSpec): Record<string, unknown> => ({ ...roi, normalized: true })
export function pickInvRoi(id: string) {
  const st = useResearch.getState()
  const a = analysisRois().find((r) => r.id === id)
  if (a) { rememberRoi(a.roi, a.name); st.setInv({ roi: a.roi, roiId: id }) }
}
export function defaultCapture(): string | null {
  const s = useStudio.getState()
  return s.activeCaptureId ?? s.captures[0]?.id ?? null
}
const captureOrThrow = (): string => {
  const id = useResearch.getState().inv.captureId ?? defaultCapture()
  if (!id) throw new Error('Take a capture first (Acquire), then select it here.')
  return id
}
const needParams = () => { if (!useResearch.getState().inv.params.length) throw new Error('Add at least one parameter.') }
const bandsArg = (inv: InverseState) => (inv.bandFrom != null && inv.bandTo != null ? { from_nm: inv.bandFrom, to_nm: inv.bandTo } : 'all' as const)

// ---------------------------------------------------------------------------------------------- unmixing
export async function runUnmix() {
  const inv = useResearch.getState().inv
  setBusy<UnmixResult>('unmix', { state: 'running', fraction: 0, message: 'unmixing…', error: null })
  try {
    const api = await loadResearchApi()
    const captureId = captureOrThrow()
    let endmembers: NonNullable<Parameters<typeof api.unmix>[0]['endmembers']> = { source: inv.endmemberSource }
    if (inv.endmemberSource === 'nmf' || inv.endmemberSource === 'vca') endmembers = { source: inv.endmemberSource, n_components: inv.nComponents }
    if (inv.endmemberSource === 'rois') {
      const ref = referenceMeta()
      const rois = useResearch.getState().inv.roiId ? [] : []
      void rois; void ref
      const all = analysisRois()
      if (all.length < 2) throw new Error('Endmembers from ROIs need at least two ROIs drawn in the Explorer.')
      // engine ROIs are in capture pixels
      const meta = await getEngine()!.getCapture(captureId)
      const px: Record<string, unknown> = {}
      for (const r of all) px[r.name] = scaleFracToPx(r.roi, meta.width, meta.height)
      endmembers = { source: 'rois', rois: px as never }
    }
    const res = await api.unmix({ capture_id: captureId, method: inv.unmixMethod, endmembers, weighting: inv.weighting, signal: 'adu_net', ...(inv.compareGt && inv.endmemberSource === 'forward_model' ? { compare_gt: { n_pixels: 60, seed: 0 } } : {}) })
    const maps: InverseState['unmixMaps'] = {}
    for (const n of [...res.names, 'rmse']) { const r = await api.unmixMap(res.unmix_id, n); maps[n] = { w: r.width, h: r.height, data: r.data } }
    useResearch.getState().setInv({ unmixMaps: maps })
    setBusy<UnmixResult>('unmix', { state: 'done', fraction: 1, result: res })
  } catch (e) { setBusy<UnmixResult>('unmix', { state: 'error', error: errText(e) }) }
}
function scaleFracToPx(r: RoiSpec, w: number, h: number): RoiSpec {
  const fx = Math.max(w - 1, 1), fy = Math.max(h - 1, 1)
  switch (r.shape) {
    case 'rect': return { shape: 'rect', x0: r.x0 * fx, y0: r.y0 * fy, x1: r.x1 * fx, y1: r.y1 * fy }
    case 'disk': return { shape: 'disk', x: r.x * fx, y: r.y * fy, radius_px: r.radius_px * fx }
    case 'ellipse': return { shape: 'ellipse', cx: r.cx * fx, cy: r.cy * fy, rx: r.rx * fx, ry: r.ry * fy }
    case 'point': return { shape: 'point', x: r.x * fx, y: r.y * fy }
    case 'polygon': return { shape: 'polygon', points: r.points.map(([x, y]) => [x * fx, y * fy] as [number, number]) }
  }
}

// ---------------------------------------------------------------------------------------------- Fisher / design
export async function runFisher() {
  const inv = useResearch.getState().inv
  setBusy<FisherResult>('fisher', { state: 'running', fraction: 0, message: 'queued…', error: null, result: null })
  useResearch.getState().setInv({ design: { state: 'idle', fraction: 0, message: '', error: null, result: null } })
  try {
    needParams()
    const api = await loadResearchApi()
    const sur = useResearch.getState().inv.surrogate.result
    const body = {
      experiment: useStudio.getState().exp, parameters: paramsSpec(inv), roi: roiDict(inv.roi), bands: bandsArg(inv),
      ...(inv.excitations ? { excitations: inv.excitations } : {}), scale: 0.25,
    }
    const r = await api.fisher(body)
    let fr: FisherResult
    if (isJob(r)) { const rid = await followJob(r.job_id, 'fisher'); fr = await api.inverseResult<FisherResult>(rid) } else fr = r
    void sur
    setBusy<FisherResult>('fisher', { state: 'done', fraction: 1, result: fr })
  } catch (e) { setBusy<FisherResult>('fisher', { state: 'error', error: errText(e) }) }
}

export async function runDesign() {
  const inv = useResearch.getState().inv
  const fr = inv.fisher.result
  if (!fr) { toast('Compute the Fisher information first', 'error'); return }
  setBusy<DesignResult>('design', { state: 'running', fraction: 0, message: 'selecting channels…', error: null })
  try {
    const api = await loadResearchApi()
    const criterion = inv.criterion === 'D' || inv.criterion === 'A' ? inv.criterion : (`target:${fr.names[Math.min(inv.targetParam, fr.names.length - 1)]}` as const)
    const r = await api.design({ fisher_id: fr.fisher_id, k: Math.min(inv.k, fr.n_channels), criterion, budget: inv.budget, ...(inv.budget === 'exposure_time' ? { total_time_s: inv.totalTimeS } : {}) })
    if (isJob(r)) throw new Error('unexpected job response')
    setBusy<DesignResult>('design', { state: 'done', fraction: 1, result: r })
  } catch (e) { setBusy<DesignResult>('design', { state: 'error', error: errText(e) }) }
}

// ---------------------------------------------------------------------------------------------- surrogate + fits
export async function buildSurrogate() {
  const inv = useResearch.getState().inv
  setBusy<SurrogateSummary>('surrogate', { state: 'running', fraction: 0, message: 'queued…', error: null, result: null })
  useResearch.getState().setInv({ fit: { state: 'idle', fraction: 0, message: '', error: null, result: null }, fitMap: { state: 'idle', fraction: 0, message: '', error: null, result: null } })
  try {
    needParams()
    const api = await loadResearchApi()
    const grid = Object.fromEntries(inv.params.map((p) => [p.name, [p.lo, p.hi]]))
    const { job_id } = await api.createSurrogate({
      experiment: useStudio.getState().exp, parameters: paramsSpec(inv), mode: 'sobol', grid, n_points: inv.gridPoints, roi: roiDict(inv.roi), scale: 0.25,
      ...(inv.excitations ? { excitations: inv.excitations } : {}), smooth: 'gcv', cache: true,
    })
    const rid = await followJob(job_id, 'surrogate')
    const sum = await api.inverseResult<SurrogateSummary>(rid)
    setBusy<SurrogateSummary>('surrogate', { state: 'done', fraction: 1, result: { ...sum, surrogate_id: sum.surrogate_id ?? rid } })
  } catch (e) { setBusy<SurrogateSummary>('surrogate', { state: 'error', error: errText(e) }) }
}

export async function runFit() {
  const inv = useResearch.getState().inv
  const sur = inv.surrogate.result
  setBusy<FitResult>('fit', { state: 'running', fraction: 0, message: 'fitting…', error: null })
  try {
    if (!sur) throw new Error('Build the surrogate first.')
    const api = await loadResearchApi()
    const r = await api.fit({ surrogate_id: sur.surrogate_id, observation: { capture_id: captureOrThrow(), roi: roiDict(inv.roi) }, profile: true, level: 0.95 })
    setBusy<FitResult>('fit', { state: 'done', fraction: 1, result: r })
  } catch (e) { setBusy<FitResult>('fit', { state: 'error', error: errText(e) }) }
}

export async function runFitMap(gtName: string | null) {
  const inv = useResearch.getState().inv
  const sur = inv.surrogate.result
  setBusy<FitMapResult>('fitMap', { state: 'running', fraction: 0, message: 'retrieving map…', error: null })
  try {
    if (!sur) throw new Error('Build the surrogate first.')
    const api = await loadResearchApi()
    const r = await api.fitMap({ surrogate_id: sur.surrogate_id, observation: { capture_id: captureOrThrow() }, map: { tile: inv.tile, ...(gtName ? { gt_name: gtName } : {}) } })
    setBusy<FitMapResult>('fitMap', { state: 'done', fraction: 1, result: r })
  } catch (e) { setBusy<FitMapResult>('fitMap', { state: 'error', error: errText(e) }) }
}

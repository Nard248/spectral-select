/** Side effects of the Research workspace: build and run a study, follow it live, load results, sensitivity, Methods text. */
import type { Experiment } from '../generated/experiment'
import type { CaptureMeta, RoiSpec } from '../api/types'
import type { MetricSpec, StudyEvent, StudyRunRow, StudySpec } from '../api/researchTypes'
import { useStudio, activeCamera, activeSource } from '../state/store'
import { useAnalysis } from '../analysis/store'
import { isRegion } from '../analysis/types'
import { runPreview } from '../state/actions'
import { ApiError } from '../api/client'
import { toast } from '../ui/toast'
import { bandCentres } from '../lib/spectral'
import { setByPointer, type ParamOption } from '../sweep/paths'
import { loadResearchApi } from './api'
import { defaultRange, ROI_PRESETS, roiToFraction, runAssignments, specProblems, suggestLinks } from './logic'
import { useResearch, type ParamDraft } from './store'

// ---------------------------------------------------------------------------------------------- experiment helpers
export interface ExpChannels { exs: number[]; bands: (ex: number) => number[] }
export function expChannels(exp: Experiment): ExpChannels {
  const src = activeSource(exp), f = activeCamera(exp).filter
  const exs = src.sequence.filter((s) => s.enabled).map((s) => s.wavelength_nm)
  return { exs, bands: (ex) => bandCentres(f.band_start_nm, f.band_step_nm, f.n_bands, f.band_start_policy, f.start_offset_nm, ex) }
}
/** The capture that defines the image grid for ROI conversion: the active capture, else the engine preview. */
export function referenceMeta(): CaptureMeta | null {
  const s = useStudio.getState()
  return s.captures.find((c) => c.id === s.activeCaptureId) ?? s.previewMeta ?? s.captures[0] ?? null
}
/** ROIs drawn in the Explorer (regions only) converted to image fractions, with the names to show in the pickers. */
export function analysisRois(): { id: string; name: string; roi: RoiSpec; color: string }[] {
  const ref = referenceMeta()
  if (!ref) return []
  return useAnalysis.getState().rois.filter((r) => isRegion(r.spec)).map((r) => ({ id: r.id, name: r.name, color: r.color, roi: roiToFraction(r.spec as RoiSpec, ref.width, ref.height) }))
}
const roiNames = new Map<string, string>()
const roiKey = (r: RoiSpec) => JSON.stringify(r)
export function rememberRoi(roi: RoiSpec, name: string) { roiNames.set(roiKey(roi), name) }
export function roiLabel(roi: RoiSpec | null | undefined): string {
  if (!roi) return 'Whole image'
  const p = ROI_PRESETS.find((x) => roiKey(x.roi) === roiKey(roi))
  return p?.label ?? roiNames.get(roiKey(roi)) ?? 'Custom'
}

// ---------------------------------------------------------------------------------------------- builder
export function paramFromOption(o: ParamOption, all: string[]): ParamDraft {
  const r = defaultRange(o.value, o.unit, o.label)
  const links = suggestLinks(o.path, all)
  return {
    pointer: o.path, label: o.label.split(' › ').slice(-2).join(' › '), kind: 'continuous', low: r.low, high: r.high, scale: r.scale,
    values: null, also_set: links, unit: o.unit, group: o.group, current: o.value,
  }
}
export function addParam(o: ParamOption, all: string[]) {
  const st = useResearch.getState()
  if (st.params.some((p) => p.pointer === o.path)) return
  let d = paramFromOption(o, all)
  const taken = new Set(st.params.map((p) => p.label))
  if (taken.has(d.label)) d = { ...d, label: o.label }
  let i = 2; const base = d.label ?? ''
  while (taken.has(d.label)) d = { ...d, label: `${base} ${i++}` }
  st.set({ params: [...st.params, d], touched: true })
}
export function patchParam(i: number, patch: Partial<ParamDraft>) {
  const st = useResearch.getState()
  st.set({ params: st.params.map((p, j) => (j === i ? { ...p, ...patch } : p)), touched: true })
}
export function removeParam(i: number) {
  const st = useResearch.getState()
  st.set({ params: st.params.filter((_, j) => j !== i), touched: true })
}

export function newMetric(exp: Experiment, kind: MetricSpec['kind'] = 'band_sum', existing: MetricSpec[] = []): MetricSpec {
  const ch = expChannels(exp)
  const ex = ch.exs[0] ?? 365
  const bs = ch.bands(ex)
  const sel = useAnalysis.getState().rois.find((r) => r.id === useAnalysis.getState().selectedId && isRegion(r.spec))
  const ar = analysisRois()
  const picked = sel ? ar.find((r) => r.id === sel.id) : undefined
  const roi = picked?.roi ?? ROI_PRESETS[1].roi
  if (picked) rememberRoi(picked.roi, picked.name)
  let name = kind === 'band_sum' ? 'signal' : kind.replace('_', ' ')
  const taken = new Set(existing.map((m) => m.name))
  const base = name; let n = 2
  while (taken.has(name)) name = `${base} ${n++}`
  const m: MetricSpec = { name, kind, roi, roi_units: 'fraction', norm: { mode: 'adu' }, stat: 'mean', contrast: 'michelson', ex }
  if (kind === 'roi_band') m.band_nm = bs[Math.floor(bs.length / 2)]
  if (kind === 'band_ratio') { m.a = { ex, band_nm: bs[Math.floor(bs.length / 3)] }; m.b = { ex, band_nm: bs[Math.floor((2 * bs.length) / 3)] } }
  if (kind === 'band_sum') { m.from_nm = bs[0]; m.to_nm = bs[bs.length - 1] }
  if (kind === 'contrast' || kind === 'cnr') { m.roi = ROI_PRESETS[1].roi; m.roi_b = ROI_PRESETS[2].roi }
  if (kind === 'gt_stat') m.gt = 'layer_thickness:cover'
  return m
}

export function buildSpec(exp: Experiment = useStudio.getState().exp): StudySpec {
  const s = useResearch.getState()
  return {
    name: s.name.trim() || 'study',
    base_experiment: exp,
    parameters: s.params.map((d) => ({ pointer: d.pointer, label: d.label ?? d.pointer, kind: d.kind, low: d.low, high: d.high, scale: d.scale, values: d.values, also_set: d.also_set })),
    design: s.design,
    replicates: s.replicates,
    mode: s.mode,
    excitations: s.excitations,
    metrics: s.metrics,
    max_runs: 20000,
  }
}
export function builderProblems(): string[] {
  const spec = buildSpec()
  return specProblems(spec)
}

// ---------------------------------------------------------------------------------------------- run
let unwatch: (() => void) | null = null
let fetching = false
let lastFetch = 0

async function fetchRunsAll(id: string): Promise<boolean> {
  const api = await loadResearchApi()
  const rows: StudyRunRow[] = []
  let columns: string[] = [], total = 0, off = 0
  try {
    for (;;) {
      const page = await api.studyRuns(id, off, 5000)
      columns = page.columns; total = page.total
      rows.push(...page.rows)
      off += page.rows.length
      if (page.rows.length === 0 || off >= page.total) break
    }
  } catch (e) {
    if (e instanceof ApiError && e.status === 409) return false
    if (/409/.test((e as Error).message)) return false
    throw e
  }
  const st = useResearch.getState()
  if (st.run?.id !== id) return false
  const ok = rows.filter((r) => r.status === 'ok' && typeof r.t_capture_s === 'number')
  const observed = { ...st.observed }
  if (ok.length >= 3 && st.run) observed[st.run.spec.mode] = ok.reduce((a, r) => a + (r.t_capture_s as number) + (typeof r.t_metrics_s === 'number' ? r.t_metrics_s : 0), 0) / ok.length
  st.set({ columns, rows, total, observed, metricSel: st.metricSel ?? (st.run?.spec.metrics[0]?.name ?? null), xParam: st.xParam ?? (st.run?.spec.parameters[0]?.label ?? null) })
  return true
}
async function refreshRows(id: string, force = false) {
  if (fetching) return
  if (!force && Date.now() - lastFetch < 1400) return
  fetching = true
  let got = true
  try { got = await fetchRunsAll(id) } catch { /* transient */ } finally { fetching = false; lastFetch = Date.now() + (got ? 0 : 5000) }  // 409 = nothing written yet: back off
}

export async function loadSensitivity(id?: string) {
  const st = useResearch.getState()
  const sid = id ?? st.run?.id
  if (!sid || !st.run) return
  const k = st.run.spec.design.kind
  if (k !== 'saltelli' && k !== 'morris') { st.set({ sens: null, sensError: null }); return }
  st.set({ sensLoading: true, sensError: null })
  try {
    const api = await loadResearchApi()
    const sens = await api.sensitivity(sid, 300)
    if (useResearch.getState().run?.id === sid) useResearch.getState().set({ sens, sensLoading: false })
  } catch (e) { useResearch.getState().set({ sens: null, sensLoading: false, sensError: (e as Error).message.replace(/^GET [^\s]+ → /, '').slice(0, 300) }) }
}

function onStudyEvent(id: string, e: StudyEvent) {
  const st = useResearch.getState()
  if (st.run?.id !== id) return
  if (e.type === 'progress') {
    st.set({ run: { ...st.run, state: 'running', fraction: e.fraction, message: e.message } })
    void refreshRows(id)
    return
  }
  if (e.type === 'done') {
    st.set({ run: { ...st.run, state: 'done', fraction: 1, message: 'done' } })
    void finish(id)
  } else if (e.type === 'cancelled') {
    st.set({ run: { ...st.run, state: 'cancelled', message: 'cancelled' } })
    void finish(id)
  } else {
    st.set({ run: { ...st.run, state: 'error', error: e.message, message: 'error' } })
    void finish(id)
  }
}
async function finish(id: string) {
  unwatch?.(); unwatch = null
  const api = await loadResearchApi()
  try {
    const status = await api.studyStatus(id)
    const st = useResearch.getState()
    if (st.run?.id === id) st.set({ run: { ...st.run, manifest: status.manifest ?? null, error: status.error ?? st.run.error, state: status.state, nRuns: status.n_runs } })
  } catch { /* status is optional */ }
  fetching = false
  await refreshRows(id, true)
  const st = useResearch.getState()
  if (st.run?.state === 'done' && (st.run.spec.design.kind === 'saltelli' || st.run.spec.design.kind === 'morris')) await loadSensitivity(id)
  void loadHistory()
}

export async function startStudy(): Promise<void> {
  const st = useResearch.getState()
  if (st.run && (st.run.state === 'running' || st.run.state === 'queued')) return
  const spec = buildSpec()
  const problems = specProblems(spec)
  if (problems.length) { toast(problems[0], 'error'); return }
  try {
    const api = await loadResearchApi()
    const r = await api.startStudy(spec, st.keepCaptures)
    unwatch?.()
    st.set({
      run: { id: r.study_id, spec, state: 'running', fraction: 0, message: 'queued', error: null, nRuns: r.n_runs, nPoints: r.n_points, startedAt: Date.now(), manifest: null },
      columns: [], rows: [], total: 0, sens: null, sensError: null, selectedRun: null, brush: {}, metricSel: spec.metrics[0]?.name ?? null, xParam: spec.parameters[0]?.label ?? null,
    })
    try { localStorage.setItem('sf.research.study', r.study_id) } catch { /* private mode */ }
    fetching = false; lastFetch = 0
    unwatch = api.watchStudy(r.study_id, (e) => onStudyEvent(r.study_id, e))
    void loadHistory()
  } catch (e) {
    const msg = (e as Error).message
    toast(`Study failed to start: ${msg.replace(/^POST [^\s]+ → \d+ /, '').slice(0, 220)}`, 'error')
  }
}

export async function cancelStudy() {
  const st = useResearch.getState()
  if (!st.run) return
  try { const api = await loadResearchApi(); await api.cancelStudy(st.run.id); toast('Cancelling after the current run…') } catch (e) { toast((e as Error).message, 'error') }
}

/** Open a study of the engine's history (also after a page reload, even while it is still running). */
export async function attachStudy(id: string) {
  const api = await loadResearchApi()
  unwatch?.(); unwatch = null
  const status = await api.studyStatus(id)
  const spec = status.manifest?.spec ?? useResearch.getState().history.find((h) => h.study_id === id)?.manifest?.spec ?? null
  const st = useResearch.getState()
  const prev = st.run?.id === id ? st.run : null
  const run = {
    id, spec: spec ?? prev?.spec ?? buildSpec(), state: status.state, fraction: status.fraction, message: status.message, error: status.error,
    nRuns: status.n_runs, nPoints: prev?.nPoints ?? 0, startedAt: Date.now(), manifest: status.manifest ?? null,
  }
  st.set({ run, columns: [], rows: [], total: 0, sens: null, sensError: null, selectedRun: null, brush: {}, metricSel: run.spec.metrics[0]?.name ?? null, xParam: run.spec.parameters[0]?.label ?? null })
  try { localStorage.setItem('sf.research.study', id) } catch { /* private mode */ }
  fetching = false; lastFetch = 0
  if (status.state === 'running' || status.state === 'queued') unwatch = api.watchStudy(id, (e) => onStudyEvent(id, e))
  await refreshRows(id, true)
  if (status.state === 'done') await loadSensitivity(id)
}

export async function loadHistory() {
  try { const api = await loadResearchApi(); useResearch.getState().set({ history: await api.listStudies() }) } catch { /* engine busy */ }
}

/** Load the builder from the spec of a finished study ("duplicate and tweak"). */
export function loadSpecIntoBuilder(spec: StudySpec) {
  useResearch.getState().set({
    name: spec.name, design: spec.design, replicates: spec.replicates, mode: spec.mode, excitations: spec.excitations ?? null, metrics: structuredClone(spec.metrics), touched: true,
    params: spec.parameters.map((p) => ({ ...p, unit: '', group: '', current: null })),
  })
  toast('Study loaded into the builder')
}

// ---------------------------------------------------------------------------------------------- results -> design
/** Apply the parameter values of one run to the experiment in the Design tab (one undo step). */
export function applyRunToDesign(row: StudyRunRow, spec: StudySpec) {
  let exp = useStudio.getState().exp
  try {
    for (const a of runAssignments(spec, row)) exp = setByPointer(exp, a.pointer, a.value)
  } catch (e) { toast(`Could not apply the run: ${(e as Error).message}`, 'error'); return false }
  useStudio.getState().setExperiment(exp, { keepHistory: true })
  useStudio.getState().setMode('design')
  toast(`Run ${row.run_id} applied to the experiment (undo with ⌘Z)`)
  return true
}
export async function previewRun(row: StudyRunRow, spec: StudySpec) {
  if (applyRunToDesign(row, spec)) await runPreview()
}

// ---------------------------------------------------------------------------------------------- methods + provenance
export async function generateMethods(style = useResearch.getState().methodsStyle, forStudy = useResearch.getState().methodsFor === 'study') {
  const st = useResearch.getState()
  st.set({ methodsBusy: true, methodsError: null })
  try {
    const api = await loadResearchApi()
    const exp = useStudio.getState().exp
    const spec = forStudy ? (st.run?.spec ?? buildSpec()) : null
    const base = spec && typeof spec.base_experiment !== 'string' ? spec.base_experiment : exp
    const r = await api.methodsText({ experiment: base as Experiment, study: spec, style })
    useResearch.getState().set({ methodsText: r.text, methodsBusy: false, methodsStyle: r.style })
  } catch (e) { useResearch.getState().set({ methodsBusy: false, methodsError: (e as Error).message.replace(/^POST [^\s]+ → /, '').slice(0, 400) }) }
}
export async function loadProvenance() {
  try { const api = await loadResearchApi(); useResearch.getState().set({ provenance: await api.provenance() }) } catch { /* optional */ }
}
export async function loadCalibrated(): Promise<void> {
  if (useResearch.getState().calibrated) return
  try { const api = await loadResearchApi(); useResearch.getState().set({ calibrated: await api.calibratedFluorophores() }) } catch { useResearch.setState({ calibrated: [] }) }
}

import { useStudio, activeSource, activeCamera } from './store'
import { detectEngine, getEngine } from '../api/client'
import { toast } from '../ui/toast'
import { describeProgress } from './progress'
import type { CaptureMeta, JobEvent } from '../api/types'

let unwatch: (() => void) | null = null

export async function initEngine(force = false) {
  const eng = await detectEngine(force)
  useStudio.setState({ engine: eng.kind === 'real' ? 'online' : 'mock' })
  try { useStudio.getState().setCaptures(await eng.listCaptures()) } catch { /* no captures yet */ }
  return eng
}

let valTimer: ReturnType<typeof setTimeout> | null = null
let valSeq = 0
/** Debounced POST /api/validate on every experiment edit. */
export function startValidation() {
  const run = async () => {
    const eng = getEngine(); if (!eng) return
    const my = ++valSeq
    try {
      const r = await eng.validate(useStudio.getState().exp)
      if (my === valSeq) useStudio.getState().setValidation(r.errors, r.warnings)
    } catch (err) {
      if (my === valSeq) useStudio.getState().setValidation([], [{ code: 'validate_failed', severity: 'warn', message: `Validation unavailable: ${(err as Error).message.slice(0, 120)}` }])
    }
  }
  const unsub = useStudio.subscribe((s, p) => {
    if (s.exp === p.exp && s.engine === p.engine) return
    if (valTimer) clearTimeout(valTimer)
    valTimer = setTimeout(run, 380)
  })
  run()
  return unsub
}

export async function runPreview() {
  const eng = getEngine() ?? (await initEngine())
  const st = useStudio.getState()
  if (st.previewing) return
  useStudio.setState({ previewing: true })
  try {
    const meta = await eng.preview({ experiment: st.exp, excitation_nm: st.previewNm, scale: st.exp.acquisition.preview_scale })
    useStudio.setState({ previewMeta: meta, previewExp: st.exp })
    toast(`Preview ready · ${meta.width}×${meta.height} · ${(meta.timings_s.total ?? 0).toFixed(2)} s`)
  } catch (err) { toast(`Preview failed: ${(err as Error).message}`, 'error') }
  finally { useStudio.setState({ previewing: false }) }
}

export async function runCapture(mode: 'sample' | 'dark' | 'white_reference' = 'sample') {
  const eng = getEngine() ?? (await initEngine())
  const st = useStudio.getState()
  if (st.job?.state === 'running') return
  const steps = activeSource(st.exp).sequence.filter((s) => s.enabled)
  const exp = { ...st.exp, acquisition: { ...st.exp.acquisition, mode } }
  try {
    const { job_id } = await eng.startCapture({ experiment: exp, mode })
    useStudio.setState({ job: { jobId: job_id, kind: mode === 'sample' ? 'capture' : mode, fraction: 0, message: 'Starting…', label: 'Starting…', startedAt: Date.now(), activeNm: steps[0]?.wavelength_nm ?? null, state: 'running' } })
    unwatch?.()
    unwatch = eng.watchJob(job_id, (ev: JobEvent) => onJobEvent(ev, steps.map((s) => s.wavelength_nm)))
  } catch (err) { toast(`Capture failed to start: ${(err as Error).message}`, 'error') }
}

function onJobEvent(ev: JobEvent, wls: number[]) {
  const set = (p: Partial<NonNullable<ReturnType<typeof useStudio.getState>['job']>>) =>
    useStudio.setState((s) => (s.job ? { job: { ...s.job, ...p } } : {}))
  if (ev.type === 'progress') {
    const info = describeProgress(ev.message, wls)
    set({ fraction: ev.fraction, message: ev.message, label: info.label, activeNm: info.activeNm ?? wls[0] ?? null })
  } else if (ev.type === 'done') {
    set({ fraction: 1, message: 'Done', state: 'done', activeNm: null })
    unwatch?.(); unwatch = null
    getEngine()?.getCapture(ev.capture_id).then((m: CaptureMeta) => {
      useStudio.getState().upsertCapture(m)
      useStudio.setState({ activeCaptureId: m.id })
      toast(`Capture "${m.name}" ready — open Explore`)
    }).catch((e: Error) => toast(`Could not fetch capture: ${e.message}`, 'error'))
    setTimeout(() => useStudio.setState((s) => (s.job?.state === 'done' ? { job: null } : {})), 2500)
  } else if (ev.type === 'error') {
    set({ state: 'error', message: ev.message, activeNm: null }); toast(`Capture error: ${ev.message}`, 'error')
    unwatch?.(); unwatch = null
    setTimeout(() => useStudio.setState((s) => (s.job?.state === 'error' ? { job: null } : {})), 4000)
  } else {
    set({ state: 'cancelled', message: 'Cancelled', activeNm: null }); unwatch?.(); unwatch = null
    setTimeout(() => useStudio.setState((s) => (s.job?.state === 'cancelled' ? { job: null } : {})), 1500)
  }
}

export async function cancelJob() {
  const st = useStudio.getState(); const eng = getEngine()
  if (!st.job?.jobId || !eng) return
  try { await eng.cancelJob(st.job.jobId) } catch (err) { toast(`Cancel failed: ${(err as Error).message}`, 'error') }
}

export const estimateSeconds = (exp: ReturnType<typeof useStudio.getState>['exp']) => {
  const src = activeSource(exp), cam = activeCamera(exp)
  const steps = src.sequence.filter((s) => s.enabled)
  // exposure per band × bands + LCTF retune overhead
  return steps.reduce((a, s) => a + (s.exposure_ms / 1000) * cam.filter.n_bands + 0.05 * cam.filter.n_bands + 0.8, 0)
}

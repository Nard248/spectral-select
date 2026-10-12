import { useEffect, useState } from 'react'
import { ChevronLeft, ChevronRight, X, Play, Pause } from 'lucide-react'
import { useStudio } from '../state/store'
import { STEPS, TOTAL, tourBackup, resetUiForTour, restoreOriginal, LAYER_STEP0, type TourCtx } from './steps'
import { useTour } from './tourStore'

const sleepFor = (token: number) => (ms: number) => new Promise<void>((res) => {
  const t0 = performance.now()
  const tick = () => { if (useTour.getState().token !== token || performance.now() - t0 >= ms) res(); else setTimeout(tick, Math.min(100, ms)) }
  tick()
})

let autoTimer: ReturnType<typeof setTimeout> | null = null

/** Run step `i`: invalidates anything in flight, executes the step's script, then (if auto-play) schedules the next step. */
export async function goToStep(i: number) {
  if (i < 0) return
  if (i >= TOTAL) { endTour(); return }
  if (autoTimer) { clearTimeout(autoTimer); autoTimer = null }
  if (i < LAYER_STEP0) restoreOriginal()   // stepping back out of the layered demo scene
  const token = useTour.getState().token + 1
  useTour.setState({ active: true, step: i, total: TOTAL, caption: null, busy: true, token, thumbs: [] })
  const ctx: TourCtx = {
    stale: () => useTour.getState().token !== token,
    sleep: sleepFor(token),
    caption: (t) => { if (useTour.getState().token === token) useTour.setState({ caption: t }) },
  }
  try { await STEPS[i].enter(ctx) } catch (err) { console.error('tour step failed', err) }
  if (useTour.getState().token !== token) return
  useTour.setState({ busy: false })
  if (useTour.getState().auto) autoTimer = setTimeout(() => { if (useTour.getState().token === token) void goToStep(i + 1) }, STEPS[i].dwell * 1000)
}

export function startTour(opts: { auto?: boolean; step?: number } = {}) {
  const s = useStudio.getState()
  if (!useTour.getState().active) { tourBackup.exp = s.exp; tourBackup.past = s.past; tourBackup.dirty = s.dirty }
  useTour.setState({ auto: !!opts.auto })
  void goToStep(opts.step ?? 0)
}

export function endTour() {
  if (autoTimer) { clearTimeout(autoTimer); autoTimer = null }
  useTour.setState((t) => ({ active: false, busy: false, caption: null, auto: false, token: t.token + 1 }))
  const s = useStudio.getState()
  // leave the user with their own experiment, restored exactly if the physics demo was interrupted
  if (tourBackup.exp && s.exp !== tourBackup.exp && s.exp.acquisition.physics.scattering !== tourBackup.exp.acquisition.physics.scattering) {
    useStudio.setState({ exp: tourBackup.exp, past: tourBackup.past, future: [], dirty: tourBackup.dirty })
  }
  useTour.setState({ thumbs: [] })
  restoreOriginal()
  tourBackup.exp = null
  resetUiForTour()
  useStudio.setState({ previewNm: null })
  s.setUi({ showFrustum: true })
}

if (import.meta.env.DEV) (window as unknown as { sfTour: unknown }).sfTour = { startTour, goToStep, endTour }

export function Tour() {
  const t = useTour()
  const exp = useStudio((s) => s.exp)
  const mode = useStudio((s) => s.mode)
  const leftOpen = useStudio((s) => s.ui.leftOpen)
  const rightOpen = useStudio((s) => s.ui.rightOpen)
  const [vw, setVw] = useState(window.innerWidth)
  useEffect(() => { const f = () => setVw(window.innerWidth); window.addEventListener('resize', f); return () => window.removeEventListener('resize', f) }, [])
  const step = STEPS[t.step]

  useEffect(() => {
    if (!t.active) return
    const h = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)) return
      if (e.key === 'ArrowRight' || e.key === 'PageDown') { e.preventDefault(); e.stopPropagation(); void goToStep(useTour.getState().step + 1) }
      else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); e.stopPropagation(); void goToStep(useTour.getState().step - 1) }
      else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); endTour() }
    }
    window.addEventListener('keydown', h, true)
    return () => window.removeEventListener('keydown', h, true)
  }, [t.active])

  useEffect(() => {
    if (!new URLSearchParams(location.search).has('tour')) return
    const q = new URLSearchParams(location.search)
    const id = setTimeout(() => startTour({ auto: q.get('tour') === 'auto', step: Number(q.get('step') ?? 0) }), 2500)
    return () => clearTimeout(id)
  }, [])

  if (!t.active || !step) return null
  // dock the card in the lower-left of the stage so it never covers the camera-view PiP (bottom right) or the sample area
  const left = mode === 'explore' ? 256 : (mode === 'design' && leftOpen ? 264 : 0) + 16
  const reserved = mode === 'explore' ? 400 + 16 : (rightOpen ? 328 : 0) + 324
  const place = { left, width: Math.max(340, Math.min(540, vw - left - reserved)), bottom: mode === 'explore' ? 16 : 100 }
  return (
    <div className="tour glass" style={place} role="dialog" aria-label="Guided tour">
      <div className="tour-head">
        <span className="tour-step mono">{t.step + 1} / {t.total}</span>
        <b>{step.title}</b>
        {t.busy && <span className="tour-busy" aria-label="working" />}
        <button className="btn ghost icon sm" style={{ marginLeft: 'auto' }} onClick={endTour} title="End tour [Esc]" aria-label="End tour"><X size={14} /></button>
      </div>
      <p>{t.caption ?? step.text(exp)}</p>
      {t.thumbs.length > 0 && <div className="tour-thumbs">{t.thumbs.map((th) => <figure key={th.label}><img src={th.url} alt={th.label} /><figcaption>{th.label}</figcaption></figure>)}</div>}
      <div className="tour-foot">
        <div className="tour-dots">{STEPS.map((s, i) => <button key={s.title} className={i === t.step ? 'on' : i < t.step ? 'done' : ''} title={s.title} onClick={() => void goToStep(i)} aria-label={s.title} />)}</div>
        <button className={`btn ghost sm${t.auto ? ' on' : ''}`} title="Auto-advance" onClick={() => {
          const auto = !useTour.getState().auto
          useTour.setState({ auto })
          if (auto && !useTour.getState().busy) { const i = useTour.getState().step; if (autoTimer) clearTimeout(autoTimer); autoTimer = setTimeout(() => void goToStep(i + 1), STEPS[i].dwell * 1000) }
          if (!auto && autoTimer) { clearTimeout(autoTimer); autoTimer = null }
        }}>{t.auto ? <Pause size={12} /> : <Play size={12} />}{t.auto ? 'Auto' : 'Auto-play'}</button>
        <button className="btn sm" disabled={t.step === 0} onClick={() => void goToStep(t.step - 1)}><ChevronLeft size={13} />Back</button>
        <button className="btn primary sm" onClick={() => void goToStep(t.step + 1)}>{t.step === t.total - 1 ? 'Finish' : <>Next<ChevronRight size={13} /></>}</button>
      </div>
    </div>
  )
}

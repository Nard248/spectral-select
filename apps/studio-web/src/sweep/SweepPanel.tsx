import { useEffect, useMemo, useRef, useState } from 'react'
import { ChevronDown, FileJson, FileSpreadsheet, Play, Search, Square, Activity } from 'lucide-react'
import { getEngine } from '../api/client'
import type { CaptureMeta, RoiSpec, SweepRequest } from '../api/types'
import { useStudio } from '../state/store'
import { useAnalysis } from '../analysis/store'
import { isRegion } from '../analysis/types'
import { scaleRoi } from '../analysis/roi'
import { metricExcitations, metricFromEem, metricLabel, metricStdFromEem, metricUnit } from '../analysis/metrics'
import { MetricPicker, defaultMetric } from '../analysis/MetricPicker'
import { NORM_MODES, normUnit } from '../analysis/norm'
import { downloadText, safeName, sweepToCsv, toJson, type SweepRow } from '../analysis/exportData'
import { WsEmpty } from '../analysis/parts'
import { XYChart, fmtVal } from '../ui/plots'
import { Num, Seg, Toggle } from '../ui/controls'
import { toast } from '../ui/toast'
import { enumerateParams, getByPointer, GROUP_ORDER, type ParamOption } from './paths'
import { parseList, rangeValues, useSweep, type SweepRun } from './store'

let unwatch: (() => void) | null = null

async function openCapture(id: string) {
  const eng = getEngine(); if (!eng) return
  try {
    const m = await eng.getCapture(id)
    useStudio.getState().upsertCapture(m)
    useStudio.setState({ activeCaptureId: id })
    useAnalysis.getState().setMid('image')
    toast(`Opened capture ${id}`)
  } catch (e) { toast(`Could not open capture: ${(e as Error).message}`, 'error') }
}

export function SweepPanel({ meta }: { meta: CaptureMeta | null }) {
  const exp = useStudio((s) => s.exp)
  const cfg = useSweep((s) => s.cfg)
  const setCfg = useSweep((s) => s.setCfg)
  const run = useSweep((s) => s.run)
  const rois = useAnalysis((s) => s.rois)
  const norm = useAnalysis((s) => s.norm)
  const stat = useAnalysis((s) => s.stat)
  const regions = useMemo(() => rois.filter((r) => isRegion(r.spec)), [rois])
  const params = useMemo(() => enumerateParams(exp), [exp])
  const sel = params.find((p) => p.path === cfg.path) ?? null
  const [pickOpen, setPickOpen] = useState(false)

  // defaults: a layer thickness if there is one (the headline sweep), else nothing
  useEffect(() => {
    if (cfg.path && sel) return
    const first = params.find((p) => p.group === 'Layer thickness' && /cover|collagen/i.test(p.layer ?? '')) ?? params.find((p) => p.group === 'Layer thickness') ?? null
    if (first) choose(first)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params, cfg.path])
  useEffect(() => { if (meta && !cfg.metric) setCfg({ metric: defaultMetric(meta) }) }, [meta, cfg.metric, setCfg])

  function choose(p: ParamOption) {
    const th = p.unit === 'mm' && /thickness/.test(p.label)
    const to = th ? Math.max(2, +(p.value * 2).toFixed(2)) : p.value === 0 ? 1 : +(p.value * 2).toPrecision(3)
    setCfg({ path: p.path, label: p.label, from: th ? 0 : +(p.value * 0.25).toPrecision(3), to, steps: th ? 9 : 8, spacing: 'linear' })
    setPickOpen(false)
  }

  const values = cfg.mode === 'list' ? parseList(cfg.list) : rangeValues(cfg.from, cfg.to, cfg.steps, cfg.spacing)
  const roi = regions.find((r) => r.id === cfg.roiId) ?? regions.find((r) => r.id === useAnalysis.getState().selectedId) ?? regions[0] ?? null
  const running = run?.state === 'running'
  const needsRefs = norm.mode === 'dark_subtracted' || norm.mode === 'white_normalized'
  const effNorm = cfg.runMode === 'preview' && needsRefs ? { mode: 'exposure_power' as const } : norm
  const needEx = cfg.metric ? metricExcitations(cfg.metric) : []
  const previewMulti = cfg.runMode === 'preview' && needEx.length > 1
  const problem = previewMulti ? `Preview sweeps capture a single excitation, but this metric reads ${needEx.join(' and ')} nm — switch to Full or pick a single-excitation metric.` : !sel ? 'Pick a parameter to sweep.' : values.length < 1 ? 'Enter at least one value.' : !cfg.metric ? 'Pick a metric.' : !meta ? 'Take a capture first — the metric and ROI are defined on it.' : !roi ? 'Draw an ROI on the image (the metric is the ROI average).' : null

  async function start() {
    const eng = getEngine()
    if (!eng || !sel || !cfg.metric || !meta || !roi) return
    unwatch?.()
    try {
      const refSize: [number, number] = [meta.width, meta.height]
      // preview captures hold ONE excitation (the first enabled step): enable only the step the metric reads
      let swExp = exp
      if (cfg.runMode === 'preview') {
        swExp = structuredClone(exp)
        const src = swExp.sources.find((q) => q.id === swExp.active_source_id) ?? swExp.sources[0]
        let any = false
        src.sequence.forEach((st) => { st.enabled = st.enabled && Math.abs(st.wavelength_nm - needEx[0]) < 0.5; any = any || st.enabled })
        if (!any) { toast(`The active source has no enabled ${needEx[0]} nm step`, 'error'); return }
      }
      const req: SweepRequest = { experiment: swExp, parameter: { path: sel.path, values, label: cfg.label || sel.label }, mode: cfg.runMode }
      const { sweep_id } = await eng.startSweep(req)
      const note = effNorm.mode !== norm.mode ? `Dark / white references do not apply to swept captures — using ${NORM_MODES.find((m) => m.value === effNorm.mode)?.label}.` : null
      const r: SweepRun = {
        id: sweep_id, path: sel.path, label: cfg.label || sel.label, values, points: [], state: 'running', fraction: 0, message: 'starting…', error: null,
        experimentName: exp.name, metric: cfg.metric, roiId: roi.id, roi: roi.spec as RoiSpec, refSize, norm: effNorm, stat, unit: normUnit(effNorm.mode), note,
      }
      useSweep.getState().startRun(r)
      unwatch = eng.watchSweep(sweep_id, (e) => {
        useSweep.getState().onEvent(e)
        if (e.type === 'point') {
          useSweep.getState().setPoint(e.index, { captureId: e.capture_id })
          void (async () => {
            try {
              const cm = await eng.getCapture(e.capture_id)
              const roiS = scaleRoi(r.roi!, cm.width / r.refSize[0], cm.height / r.refSize[1])
              const res = await eng.eem(e.capture_id, { roi: roiS, norm: r.norm, stat: r.stat })
              const v = metricFromEem(res, r.metric!)
              const ok = Number.isFinite(v)
              useSweep.getState().setPoint(e.index, { metric: ok ? v : null, std: metricStdFromEem(res, r.metric!), err: ok ? null : `capture has no ${metricLabel(r.metric!)} (excitations ${res.excitations_nm.join(', ')} nm)` })
              if (res.unit) useSweep.getState().patchRun({ unit: res.unit })
            } catch (err) { useSweep.getState().setPoint(e.index, { err: (err as Error).message }); useSweep.getState().patchRun({ error: (err as Error).message }) }
          })()
        }
      })
    } catch (err) { toast(`Sweep failed to start: ${(err as Error).message}`, 'error') }
  }
  async function cancel() {
    const eng = getEngine(); if (!eng || !run) return
    try { await eng.cancelSweep(run.id) } catch { /* already finished */ }
  }

  return (
    <div className="sw">
      <div className="cfg">
        <div className="sec-h">Parameter</div>
        <div style={{ padding: '0 12px', position: 'relative' }}>
          <button className="pickbtn" onClick={() => setPickOpen(!pickOpen)} aria-haspopup="listbox" aria-expanded={pickOpen}>
            <div style={{ minWidth: 0 }}><b>{sel?.label ?? 'Choose a parameter…'}</b><small>{sel ? `${sel.path}` : 'layer thickness, dye concentration, scattering, power, camera…'}</small></div><ChevronDown size={14} />
          </button>
          {pickOpen && <ParamPicker params={params} value={cfg.path} onPick={choose} onClose={() => setPickOpen(false)} />}
          {sel && <div className="hint" style={{ padding: '6px 2px 0' }}>current value <span className="mono" style={{ color: 'var(--text)' }}>{String(getByPointer(exp, sel.path))}{sel.unit ? ` ${sel.unit}` : ''}</span></div>}
        </div>

        <div className="sec-h">Values</div>
        <div style={{ padding: '0 12px', display: 'grid', gap: 8 }}>
          <Seg small value={cfg.mode} onChange={(mode) => setCfg({ mode })} options={[{ value: 'range', label: 'Range' }, { value: 'list', label: 'List' }]} />
          {cfg.mode === 'range' ? <>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 74px', gap: 6 }}>
              <div><div className="hint">from</div><Num value={cfg.from} step={0.1} bar={false} unit={sel?.unit} onChange={(from) => setCfg({ from })} /></div>
              <div><div className="hint">to</div><Num value={cfg.to} step={0.1} bar={false} unit={sel?.unit} onChange={(to) => setCfg({ to })} /></div>
              <div><div className="hint">steps</div><Num value={cfg.steps} min={1} max={60} step={1} integer bar={false} onChange={(steps) => setCfg({ steps })} /></div>
            </div>
            <Toggle value={cfg.spacing === 'log'} label="logarithmic spacing" onChange={(l) => setCfg({ spacing: l ? 'log' : 'linear' })} />
          </> : <input className="field mono" value={cfg.list} placeholder="0, 0.25, 0.5, 1, 2" spellCheck={false} onChange={(e) => setCfg({ list: e.target.value })} onKeyDown={(e) => e.stopPropagation()} />}
          <div className="hint">{values.length} point{values.length === 1 ? '' : 's'}: <span className="mono">{values.slice(0, 8).map((v) => fmtVal(v)).join(', ')}{values.length > 8 ? ' …' : ''}</span></div>
        </div>

        <div className="sec-h">Engine</div>
        <div style={{ padding: '0 12px', display: 'grid', gap: 6 }}>
          <Seg small value={cfg.runMode} onChange={(runMode) => setCfg({ runMode })} options={[{ value: 'preview', label: 'Preview', title: 'Reduced resolution and photon counts — seconds per point' }, { value: 'sample', label: 'Full', title: 'Full-quality capture per point' }]} />
          <div className="hint">{cfg.runMode === 'preview' ? 'Each point is a reduced-quality capture: fast, good for finding the trend.' : 'Each point is a full capture with the experiment’s MC quality: slow, publication numbers.'}</div>
        </div>

        <div className="sec-h">Result metric</div>
        <div style={{ padding: '0 0 0', display: 'grid', gap: 6 }}>
          <div style={{ padding: '0 12px' }}>
            <div className="hint" style={{ paddingBottom: 3 }}>ROI (the metric is averaged over it)</div>
            <select className="field" value={roi?.id ?? ''} onChange={(e) => setCfg({ roiId: e.target.value || null })}>
              {regions.length === 0 && <option value="">— draw an ROI first —</option>}
              {regions.map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}
            </select>
          </div>
          {meta && cfg.metric && <MetricPicker meta={meta} value={cfg.metric} onChange={(metric) => setCfg({ metric })} />}
        </div>

        <div className="runbar">
          {problem && <div className="hint" style={{ color: 'var(--warn)' }}>{problem}</div>}
          {running
            ? <button className="btn danger" onClick={cancel}><Square size={13} />Cancel sweep</button>
            : <button className="btn primary" disabled={!!problem} onClick={start}><Play size={14} />Run sweep · {values.length} point{values.length === 1 ? '' : 's'}</button>}
          {run && <div className="progress" aria-label="sweep progress"><i style={{ width: `${Math.round(run.fraction * 100)}%` }} /></div>}
          {run && <div className="hint">{run.state === 'running' ? run.message : run.state === 'done' ? `Done · ${run.points.length} captures` : run.state === 'error' ? `Error: ${run.error}` : 'Cancelled'}</div>}
        </div>
      </div>

      <div className="res"><SweepResults /></div>
    </div>
  )
}

export function ParamPicker({ params, value, onPick, onClose }: { params: ParamOption[]; value: string | null; onPick: (p: ParamOption) => void; onClose: () => void }) {
  const [q, setQ] = useState('')
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) onClose() }
    const k = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('mousedown', h); window.addEventListener('keydown', k)
    return () => { window.removeEventListener('mousedown', h); window.removeEventListener('keydown', k) }
  }, [onClose])
  const groups = useMemo(() => {
    const needle = q.trim().toLowerCase()
    const hit = params.filter((p) => !needle || `${p.label} ${p.group} ${p.path}`.toLowerCase().includes(needle))
    return GROUP_ORDER.map((g) => ({ g, items: hit.filter((p) => p.group === g) })).filter((x) => x.items.length)
  }, [params, q])
  return (
    <div className="ppop" ref={ref} role="listbox">
      <div style={{ position: 'relative' }}>
        <Search size={13} style={{ position: 'absolute', left: 18, top: 16, color: 'var(--faint)' }} />
        <input className="field srch" autoFocus placeholder="Search thickness, concentration, power…" value={q} style={{ paddingLeft: 30 }} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.stopPropagation()} />
      </div>
      <div className="list">
        {groups.map(({ g, items }) => <div key={g}><div className="grp">{g}</div>
          {items.map((p) => <div key={p.path} role="option" aria-selected={p.path === value} className={`opt${p.path === value ? ' on' : ''}`} onClick={() => onPick(p)} title={p.path}><span>{p.label}</span><code>{fmtVal(p.value)}{p.unit ? ` ${p.unit}` : ''}</code></div>)}</div>)}
        {!groups.length && <div className="hint" style={{ padding: 16, textAlign: 'center' }}>No parameter matches “{q}”.</div>}
      </div>
    </div>
  )
}

const shortLabel = (l: string) => l.split(' › ').slice(-2).join(' › ')

function SweepResults() {
  const run = useSweep((s) => s.run)
  const clear = useSweep((s) => s.clear)
  const [selIdx, setSel] = useState<number | null>(null)
  if (!run) return (
    <WsEmpty icon="sweep" title="Sweep a parameter, watch the signal respond">
      Choose a parameter on the left — for example the <b style={{ display: 'inline', color: 'var(--accent)' }}>cover thickness</b> from 0 to 2 mm — pick the ROI and metric to track, and run.
      Each point is simulated with the full physics; the plot fills in as results arrive and every point opens its capture.
    </WsEmpty>)
  const pts = run.points.filter((p) => p.metric != null && typeof p.value === 'number')
  const unit = run.metric ? metricUnit(run.metric, run.unit) : run.unit
  const title = run.metric ? metricLabel(run.metric) : 'metric'
  const rows: (SweepRow & { err?: string | null })[] = run.values.map((v, i) => { const p = run.points.find((q) => q.index === i); return { index: i, value: v, metric: p?.metric ?? null, std: p?.std ?? null, captureId: p?.captureId ?? null, err: p?.err ?? null } })
  const ctx = { captureId: rows.find((r) => r.captureId)?.captureId ?? '', experimentName: run.experimentName, norm: run.norm, roi: run.roi, extra: { sweep_id: run.id, parameter: run.label } }
  const stem = safeName(`${run.experimentName}_sweep_${run.path.split('/').slice(-2).join('_')}`)
  return (
    <>
      <div className="lf-card">
        <header>{title} <small>vs {shortLabel(run.label)}</small><span className="sp" />
          <button className="btn sm" disabled={!rows.some((r) => r.metric != null)} onClick={() => downloadText(`${stem}.csv`, sweepToCsv(rows, { parameterPath: run.path, parameterLabel: run.label, metricName: title, unit }, ctx))}><FileSpreadsheet size={13} />CSV</button>
          <button className="btn sm" disabled={!rows.some((r) => r.metric != null)} onClick={() => downloadText(`${stem}.json`, toJson('sweep', ctx, { parameter: { path: run.path, label: run.label, values: run.values }, metric: run.metric, unit, points: rows }), 'application/json')}><FileJson size={13} />JSON</button>
          <button className="btn ghost sm" onClick={clear}>Clear</button>
        </header>
        <div className="cbody">
          {run.note && <div className="note warn" style={{ margin: '0 0 8px' }}>{run.note}</div>}
          {run.state !== 'running' && rows.length > 0 && rows.every((r) => r.captureId && r.metric == null && r.err) && <div className="note err" style={{ margin: '0 0 8px' }}>No metric could be computed: {rows.find((r) => r.err)?.err}</div>}
          {run.error && run.state !== 'error' && <div className="note err" style={{ margin: '0 0 8px' }}>{run.error}</div>}
          {run.state === 'error' && <div className="note err" style={{ margin: '0 0 8px' }}>{run.error}</div>}
          <XYChart height={300} zeroBase={run.metric?.kind !== 'peak_wavelength' && run.metric?.kind !== 'band_ratio'}
            series={[{ name: title, x: pts.map((p) => p.value as number), y: pts.map((p) => p.metric as number), color: '#35d8ff', width: 2.4, points: true }]}
            xLabel={shortLabel(run.label)} yLabel={unit}
            xDomain={run.values.length > 1 && run.values.every((v) => typeof v === 'number') ? [Math.min(...(run.values as number[])), Math.max(...(run.values as number[]))] : undefined}
            tipTitle={(x) => `${run.label.split('›').pop()?.trim()} = ${fmtVal(x)}`}
            onClickPoint={(_s, i) => { const p = pts[i]; if (p?.captureId) { setSel(p.index); void openCapture(p.captureId) } }} />
          <div className="hint" style={{ paddingTop: 6 }}><Activity size={11} style={{ verticalAlign: -1 }} /> Click a point to open its capture in the image view.</div>
        </div>
      </div>
      <div className="lf-card">
        <header>Points <small>{run.values.length}</small></header>
        <div className="cbody"><div className="pointlist">
          {rows.map((r) => (
            <div key={r.index} className={`pointrow${selIdx === r.index ? ' sel' : ''}`} onClick={() => { if (r.captureId) { setSel(r.index); void openCapture(r.captureId) } }} style={{ cursor: r.captureId ? 'pointer' : 'default' }}>
              <span className={`st ${r.metric != null ? 'ok' : r.index === run.points.length && run.state === 'running' ? 'run' : ''}`} />
              <span>{fmtVal(r.value as number)}</span>
              <span style={{ color: r.metric != null ? 'var(--text)' : r.err ? 'var(--warn)' : 'var(--faint)' }} title={r.err ?? undefined}>{r.metric != null ? `${fmtVal(r.metric)}${unit ? ` ${unit}` : ''}` : r.err ? 'n/a' : '…'}</span>
              <span style={{ color: 'var(--faint)' }}>{r.captureId ?? ''}</span>
            </div>))}
        </div></div>
      </div>
    </>
  )
}

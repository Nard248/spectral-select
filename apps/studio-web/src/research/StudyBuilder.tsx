import { useEffect, useMemo, useState } from 'react'
import { Link2, Play, Plus, Square, X, Layers, History as HistoryIcon } from 'lucide-react'
import { useStudio } from '../state/store'
import { useAnalysis } from '../analysis/store'
import { enumerateParams, type ParamOption } from '../sweep/paths'
import { ParamPicker } from '../sweep/SweepPanel'
import { Num, Sel, Seg, Tip, Toggle } from '../ui/controls'
import { NORM_MODES } from '../analysis/norm'
import { fmtVal } from '../ui/plots'
import type { DesignKind, MetricKind, MetricSpec, StudyMode } from '../api/researchTypes'
import type { RoiSpec } from '../api/types'
import { wavelengthToHex } from '../lib/color'
import {
  DESIGN_INFO, DESIGN_ORDER, METRIC_KIND_INFO, METRIC_ORDER, ROI_PRESETS, designSize, estimateRuntime, fmtDuration, metricProblem, parameterValid, roiSummary, specProblems,
} from './logic'
import { useResearch, type ParamDraft } from './store'
import {
  addParam, analysisRois, attachStudy, buildSpec, cancelStudy, expChannels, newMetric, patchParam, referenceMeta, removeParam, rememberRoi, roiLabel, startStudy,
} from './actions'

const MODE_TIPS: Record<StudyMode, { label: string; tip: string }> = {
  preview: { label: 'Preview', tip: 'First excitation only, at the preview resolution. Seconds per run: use it to screen many parameters, never for metrics that read several excitations.' },
  fast: { label: 'Fast', tip: 'All excitations at the preview resolution. The default: every metric is defined, a run costs about (preview time × number of excitations).' },
  full: { label: 'Full', tip: 'All excitations at full resolution and full Monte Carlo photon counts. Publication numbers; minutes per run.' },
}

function Sec({ title, extra, children }: { title: string; extra?: React.ReactNode; children: React.ReactNode }) {
  return <><div className="sec-h">{title}{extra && <span style={{ marginLeft: 'auto', order: 3 }}>{extra}</span>}</div><div className="rs-sec">{children}</div></>
}

// ---------------------------------------------------------------------------------------------- parameters
function ParamCard({ p, i, all }: { p: ParamDraft; i: number; all: ParamOption[] }) {
  const err = parameterValid(p)
  const links = useMemo(() => all.filter((o) => o.path !== p.pointer && o.unit === p.unit && !p.also_set.includes(o.path)).slice(0, 80), [all, p])
  const [choiceText, setChoiceText] = useState<string | null>(null)
  const label = (ptr: string) => all.find((o) => o.path === ptr)?.label.split(' › ').slice(-2).join(' › ') ?? ptr
  const kindOpt = [{ value: 'continuous', label: 'Range' }, { value: 'integer', label: 'Integer' }, { value: 'choice', label: 'Choice' }] as const
  return (
    <div className={`rs-card${err ? ' bad' : ''}`}>
      <div className="rs-card-h">
        <input className="rs-name" value={p.label ?? ''} aria-label="Parameter label" spellCheck={false} onChange={(e) => patchParam(i, { label: e.target.value })} onKeyDown={(e) => e.stopPropagation()} />
        <button className="btn ghost icon sm" aria-label={`Remove ${p.label}`} onClick={() => removeParam(i)}><X size={13} /></button>
      </div>
      <div className="rs-ptr" title={p.pointer}>{p.pointer}</div>
      <Seg small value={p.kind} onChange={(kind) => patchParam(i, kind === 'choice' ? { kind, values: p.values?.length ? p.values : [p.current ?? 0, (p.current ?? 0) * 2 || 1] } : { kind, ...(p.low == null ? { low: 0, high: 1 } : {}) })} options={kindOpt.map((o) => ({ ...o }))} />
      {p.kind !== 'choice'
        ? <div className="rs-lohi">
          <div><div className="hint">low</div><Num value={p.low ?? 0} step={Math.abs((p.high ?? 1) - (p.low ?? 0)) / 50 || 0.01} bar={false} unit={p.unit} error={!!err} integer={p.kind === 'integer'} onChange={(low) => patchParam(i, { low })} /></div>
          <div><div className="hint">high</div><Num value={p.high ?? 1} step={Math.abs((p.high ?? 1) - (p.low ?? 0)) / 50 || 0.01} bar={false} unit={p.unit} error={!!err} integer={p.kind === 'integer'} onChange={(high) => patchParam(i, { high })} /></div>
          {p.kind === 'continuous' && <Toggle value={p.scale === 'log'} label="log" tip="Sample uniformly in log10 of the value (for quantities spanning decades, e.g. concentrations). Needs low > 0." onChange={(l) => patchParam(i, { scale: l ? 'log' : 'linear' })} />}
        </div>
        : <input className="field mono" spellCheck={false} placeholder="values, comma separated" value={choiceText ?? (p.values ?? []).join(', ')}
          onChange={(e) => { setChoiceText(e.target.value); const v = e.target.value.split(',').map((s) => s.trim()).filter(Boolean).map((s) => (Number.isFinite(Number(s)) ? Number(s) : s)); patchParam(i, { values: v }) }}
          onBlur={() => setChoiceText(null)} onKeyDown={(e) => e.stopPropagation()} />}
      {err && <div className="err">{err}</div>}
      <div className="rs-links">
        {p.also_set.map((l) => <span key={l} className="rs-link" title={`Receives the same value: ${l}`}><Link2 size={10} />{label(l)}<button aria-label="Unlink" onClick={() => patchParam(i, { also_set: p.also_set.filter((x) => x !== l) })}><X size={10} /></button></span>)}
        {links.length > 0 && <select className="rs-linksel" aria-label="Link another pointer" value="" onChange={(e) => { if (e.target.value) patchParam(i, { also_set: [...p.also_set, e.target.value] }) }}>
          <option value="">+ link…</option>{links.map((o) => <option key={o.path} value={o.path}>{o.label.split(' › ').slice(-2).join(' › ')}</option>)}
        </select>}
      </div>
      {p.current != null && <div className="hint">current <span className="mono" style={{ color: 'var(--text)' }}>{fmtVal(p.current)}{p.unit ? ` ${p.unit}` : ''}</span></div>}
    </div>
  )
}

function Parameters() {
  const exp = useStudio((s) => s.exp)
  const params = useResearch((s) => s.params)
  const touched = useResearch((s) => s.touched)
  const [open, setOpen] = useState(false)
  const all = useMemo(() => enumerateParams(exp), [exp])
  const paths = useMemo(() => all.map((o) => o.path), [all])
  // first visit: pre-fill the headline parameters of the scene (a layer thickness, then a dye concentration)
  useEffect(() => {
    if (touched || params.length) return
    const th = all.find((p) => p.group === 'Layer thickness' && /cover|collagen/i.test(p.layer ?? '')) ?? all.find((p) => p.group === 'Layer thickness')
    // the dye of the deepest layer that carries one (the buried fluorophore), else any dye in the scene
    const matIdx = new Map(exp.materials.map((m, i) => [m.id, i] as const))
    let dye: ParamOption | undefined
    for (const o of exp.objects) {
      const sh = o.shape as unknown as { kind: string; layers?: { material_id: string }[] }
      if (sh.kind !== 'layered_block' || !sh.layers) continue
      for (const l of [...sh.layers].reverse()) {
        const mi = matIdx.get(l.material_id)
        if (mi == null) continue
        dye = all.find((p) => p.path.startsWith(`/materials/${mi}/dyes/`) && /concentration/.test(p.label) && p.value > 0)
        if (dye) break
      }
      if (dye) break
    }
    dye ??= all.find((p) => p.group === 'Material dyes' && /concentration/.test(p.label) && p.value > 0)
    const first = [th, dye].filter(Boolean) as ParamOption[]
    first.forEach((o) => addParam(o, paths))
    useResearch.getState().set({ touched: false })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [all])
  return (
    <Sec title="Parameters" extra={<span className="tag">{params.length}</span>}>
      {params.map((p, i) => <ParamCard key={p.pointer} p={p} i={i} all={all} />)}
      {params.length === 0 && <div className="hint">Pick the experiment parameters to vary. Any numeric field of the design can be a parameter.</div>}
      <div style={{ position: 'relative' }}>
        <button className="btn sm" style={{ width: '100%' }} onClick={() => setOpen(!open)} aria-haspopup="listbox" aria-expanded={open}><Plus size={13} />Add parameter…</button>
        {open && <ParamPicker params={all.filter((o) => !params.some((p) => p.pointer === o.path))} value={null} onPick={(o) => { addParam(o, paths); setOpen(false) }} onClose={() => setOpen(false)} />}
      </div>
    </Sec>
  )
}

// ---------------------------------------------------------------------------------------------- design
function DesignSection() {
  const exp = useStudio((s) => s.exp)
  const { params, design, replicates, mode, excitations, keepCaptures, observed } = useResearch()
  const set = useResearch((s) => s.set)
  const preview = useStudio((s) => s.previewMeta)
  const ch = expChannels(exp)
  const sz = designSize(design, params, replicates)
  const prevS = preview ? (preview.timings_s.total ?? Object.values(preview.timings_s).reduce((a, b) => a + b, 0)) : null
  const nEx = excitations?.length ?? ch.exs.length
  const est = estimateRuntime(sz.runs, mode, { previewS: prevS, observed, nExcitations: nEx, previewScale: exp.acquisition.preview_scale })
  const k = params.length
  const nLabel = design.kind === 'saltelli' ? 'base samples n' : design.kind === 'morris' ? 'trajectories' : 'points'
  const setD = (p: Partial<typeof design>) => set({ design: { ...design, ...p } })
  return (
    <Sec title="Design">
      <Sel value={design.kind} onChange={(kind: DesignKind) => setD({ kind })} options={DESIGN_ORDER.map((d) => ({ value: d, label: DESIGN_INFO[d].label }))} />
      <div className="hint">{DESIGN_INFO[design.kind].blurb}</div>
      <div className="rs-grid3">
        {design.kind !== 'grid' && <div><div className="hint">{nLabel}</div><Num value={design.n} min={1} max={100000} step={1} integer bar={false} onChange={(n) => setD({ n })} /></div>}
        {(design.kind === 'grid' || design.kind === 'morris') && <div><div className="hint">levels</div><Num value={design.levels} min={2} max={64} step={design.kind === 'morris' ? 2 : 1} integer bar={false} onChange={(levels) => setD({ levels })} /></div>}
        <div><div className="hint">replicates</div><Num value={replicates} min={1} max={64} integer bar={false} onChange={(v) => set({ replicates: v })} /></div>
        <div><div className="hint">seed</div><Num value={design.seed} min={0} max={1e9} integer bar={false} onChange={(seed) => setD({ seed })} /></div>
      </div>
      <div className="rs-tiles">
        <Tip text={design.kind === 'saltelli' ? `n (k + 2) = ${design.n} × ${k + 2}` : design.kind === 'morris' ? `n (k + 1) = ${design.n} × ${k + 1}` : 'Distinct parameter combinations'}><div><b>{sz.points.toLocaleString()}</b><span>design points</span></div></Tip>
        <Tip text={`${sz.points.toLocaleString()} points × ${replicates} replicate${replicates === 1 ? '' : 's'} (each replicate has its own Monte Carlo seed)`}><div><b>{sz.runs.toLocaleString()}</b><span>runs</span></div></Tip>
        <Tip title="Estimated runtime" text={est.basis === 'observed' ? 'From the measured time per run of your last study in this mode.' : est.basis === 'preview-scaled' ? 'Scaled from the timing of your last engine preview (take one in Design with Space). Replace by a measured value after the first study.' : 'No timing available yet: a default of 1.5 s per preview capture. Take a preview to calibrate.'}><div className={est.basis === 'default' ? 'dim' : ''}><b>{fmtDuration(est.seconds)}</b><span>{est.basis === 'observed' ? 'measured' : 'estimated'}</span></div></Tip>
      </div>
      <div className="row stack" style={{ marginTop: 2 }}>
        <div className="rs-modehead"><label>Capture mode</label></div>
        <Seg small value={mode} onChange={(m: StudyMode) => set({ mode: m })} options={(['preview', 'fast', 'full'] as StudyMode[]).map((m) => ({ value: m, label: MODE_TIPS[m].label, title: MODE_TIPS[m].tip }))} />
        <div className="hint">{MODE_TIPS[mode].tip}</div>
      </div>
      {ch.exs.length > 1 && (
        <div>
          <div className="hint" style={{ paddingBottom: 3 }}>Excitations <span style={{ color: 'var(--dim)' }}>{excitations ? `${excitations.length} of ${ch.exs.length}` : 'all'}</span></div>
          <div className="rs-exs">
            {ch.exs.map((e) => {
              const on = !excitations || excitations.includes(e)
              return <button key={e} className={`rs-ex${on ? ' on' : ''}`} style={{ ['--c' as string]: wavelengthToHex(e) }} aria-pressed={on}
                onClick={() => { const cur = excitations ?? ch.exs; const next = on ? cur.filter((x) => x !== e) : [...cur, e].sort((a, b) => a - b); set({ excitations: next.length === 0 || next.length === ch.exs.length ? null : next }) }}>{e}</button>
            })}
          </div>
        </div>
      )}
      <Toggle value={keepCaptures} onChange={(v) => set({ keepCaptures: v })} label="Keep every capture (npz on the engine)" tip="Stores each run's full cube next to the results (captures/<run_id>.npz in the study folder). Large: leave off for big studies." />
    </Sec>
  )
}

// ---------------------------------------------------------------------------------------------- metrics
const STATS = ['mean', 'median', 'std', 'min', 'max'] as const

export function RoiPick({ value, onChange, label }: { value: RoiSpec | null | undefined; onChange: (r: RoiSpec) => void; label: string }) {
  const rois = useAnalysis((s) => s.rois)
  const hasRef = !!referenceMeta()
  const list = useMemo(() => (hasRef ? analysisRois() : []), [hasRef, rois])
  const cur = roiLabel(value)
  const isCustom = cur === 'Custom'
  const r = value && value.shape === 'rect' ? value : null
  const setRect = (k: 'x0' | 'y0' | 'x1' | 'y1', v: number) => { if (r) onChange({ ...r, [k]: v }) }
  return (
    <div className="rs-roi">
      <div className="hint">{label}</div>
      <select className="field" aria-label={label} value={isCustom ? '__custom' : cur} onChange={(e) => {
        const v = e.target.value
        if (v === '__custom') { onChange(r ?? ROI_PRESETS[1].roi); return }
        const p = ROI_PRESETS.find((x) => x.label === v); if (p) { onChange(p.roi); return }
        const a = list.find((x) => x.name === v); if (a) { rememberRoi(a.roi, a.name); onChange(a.roi) }
      }}>
        {ROI_PRESETS.map((p) => <option key={p.id} value={p.label}>{p.label}</option>)}
        {list.length > 0 && <optgroup label="From the Explorer">{list.map((a) => <option key={a.id} value={a.name}>{a.name}</option>)}</optgroup>}
        <option value="__custom">Custom rectangle…</option>
      </select>
      {isCustom && r && <div className="rs-grid4">{(['x0', 'y0', 'x1', 'y1'] as const).map((k) => <Num key={k} value={r[k]} min={0} max={1} step={0.05} precision={2} bar={false} onChange={(v) => setRect(k, v)} />)}</div>}
      {!isCustom && <div className="hint mono" style={{ fontSize: 10 }}>{roiSummary(value)}</div>}
      {!hasRef && <div className="hint">Draw ROIs in the Explorer after a capture to pick them here.</div>}
    </div>
  )
}

function MetricCard({ m, i }: { m: MetricSpec; i: number }) {
  const exp = useStudio((s) => s.exp)
  const captures = useStudio((s) => s.captures)
  const activeId = useStudio((s) => s.activeCaptureId)
  const metrics = useResearch((s) => s.metrics)
  const set = useResearch((s) => s.set)
  const ch = expChannels(exp)
  const patch = (p: Partial<MetricSpec>) => set({ metrics: metrics.map((x, j) => (j === i ? { ...x, ...p } : x)) })
  const ex = m.ex ?? ch.exs[0]
  const bands = ch.bands(ex)
  const nearest = (arr: number[], v: number | null | undefined) => (v == null ? arr[0] : arr.reduce((b, e) => (Math.abs(e - v) < Math.abs(b - v) ? e : b), arr[0]))
  const err = metricProblem(m)
  const meta = captures.find((c) => c.id === activeId)
  const gts = useMemo(() => (meta?.ground_truth_layers ?? []).filter((n) => !['material_index', 'object_index', 'material', 'object', 'block_face'].includes(n)), [meta])
  const exSel = (v: number, f: (e: number) => void) => <Sel value={v} options={ch.exs.map((e) => ({ value: e, label: `ex ${e} nm` }))} onChange={f} />
  const bandSel = (e: number, v: number | null | undefined, f: (b: number) => void) => <Sel value={nearest(ch.bands(e), v)} options={ch.bands(e).map((b) => ({ value: b, label: `${b.toFixed(0)} nm` }))} onChange={f} />
  const kind = m.kind
  return (
    <div className={`rs-card${err ? ' bad' : ''}`}>
      <div className="rs-card-h">
        <input className="rs-name" value={m.name} aria-label="Metric name" spellCheck={false} onChange={(e) => patch({ name: e.target.value })} onKeyDown={(e) => e.stopPropagation()} />
        <button className="btn ghost icon sm" aria-label={`Remove ${m.name}`} onClick={() => set({ metrics: metrics.filter((_, j) => j !== i) })}><X size={13} /></button>
      </div>
      <Tip text={METRIC_KIND_INFO[kind].hint} block><Sel value={kind} options={METRIC_ORDER.map((k) => ({ value: k, label: METRIC_KIND_INFO[k].label }))}
        onChange={(k: MetricKind) => { const nm = newMetric(exp, k, metrics.filter((_, j) => j !== i)); set({ metrics: metrics.map((x, j) => (j === i ? { ...nm, name: x.name, roi: x.roi ?? nm.roi } : x)) }) }} /></Tip>
      {kind !== 'band_ratio' && kind !== 'gt_stat' && kind !== 'eem_integral' && <div className="rs-two">{exSel(ex, (e) => patch({ ex: e, ...(kind === 'roi_band' ? { band_nm: nearest(ch.bands(e), m.band_nm) } : {}), ...(kind === 'band_sum' ? { from_nm: ch.bands(e)[0], to_nm: ch.bands(e).slice(-1)[0] } : {}) }))}
        {kind === 'roi_band' && bandSel(ex, m.band_nm, (band_nm) => patch({ band_nm }))}</div>}
      {kind === 'band_sum' && <div className="rs-two">{bandSel(ex, m.from_nm ?? bands[0], (from_nm) => patch({ from_nm }))}{bandSel(ex, m.to_nm ?? bands[bands.length - 1], (to_nm) => patch({ to_nm }))}</div>}
      {kind === 'band_ratio' && <>
        <div className="rs-ab"><label>A</label><div className="rs-two">{exSel(m.a?.ex ?? ex, (e) => patch({ a: { ex: e, band_nm: nearest(ch.bands(e), m.a?.band_nm) } }))}{bandSel(m.a?.ex ?? ex, m.a?.band_nm, (band_nm) => patch({ a: { ex: m.a?.ex ?? ex, band_nm } }))}</div></div>
        <div className="rs-ab"><label>B</label><div className="rs-two">{exSel(m.b?.ex ?? ex, (e) => patch({ b: { ex: e, band_nm: nearest(ch.bands(e), m.b?.band_nm) } }))}{bandSel(m.b?.ex ?? ex, m.b?.band_nm, (band_nm) => patch({ b: { ex: m.b?.ex ?? ex, band_nm } }))}</div></div></>}
      {kind === 'gt_stat' && (gts.length
        ? <Sel value={m.gt ?? gts[0]} options={gts.map((g) => ({ value: g, label: g }))} onChange={(gt) => patch({ gt })} />
        : <input className="field mono" spellCheck={false} placeholder="ground-truth map, e.g. layer_thickness:cover" value={m.gt ?? ''} onChange={(e) => patch({ gt: e.target.value })} onKeyDown={(e) => e.stopPropagation()} />)}
      <RoiPick label={kind === 'contrast' || kind === 'cnr' ? 'ROI A' : 'ROI'} value={m.roi} onChange={(roi) => patch({ roi })} />
      {(kind === 'contrast' || kind === 'cnr') && <RoiPick label="ROI B" value={m.roi_b} onChange={(roi_b) => patch({ roi_b })} />}
      <div className="rs-two">
        {kind !== 'peak_wavelength' && kind !== 'eem_integral' && kind !== 'cnr' ? <Sel value={m.stat} options={STATS.map((s) => ({ value: s, label: s }))} onChange={(stat) => patch({ stat })} /> : <span />}
        {kind === 'contrast' ? <Sel value={m.contrast} options={[{ value: 'michelson', label: 'Michelson' }, { value: 'weber', label: 'Weber' }]} onChange={(contrast) => patch({ contrast })} /> : <Sel value={m.norm.mode} options={NORM_MODES.filter((n) => !['dark_subtracted', 'white_normalized'].includes(n.value)).map((n) => ({ value: n.value, label: n.label }))} onChange={(mode) => patch({ norm: { mode } })} />}
      </div>
      {err && <div className="err">{err}</div>}
    </div>
  )
}

function Metrics() {
  const exp = useStudio((s) => s.exp)
  const metrics = useResearch((s) => s.metrics)
  const set = useResearch((s) => s.set)
  useEffect(() => { if (!metrics.length && !useResearch.getState().touched) set({ metrics: [newMetric(exp)] }) /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [])
  return (
    <Sec title="Metrics" extra={<span className="tag">{metrics.length}</span>}>
      {metrics.map((m, i) => <MetricCard key={i} m={m} i={i} />)}
      <button className="btn sm" style={{ width: '100%' }} onClick={() => set({ metrics: [...metrics, newMetric(exp, 'band_sum', metrics)], touched: true })}><Plus size={13} />Add metric</button>
    </Sec>
  )
}

// ---------------------------------------------------------------------------------------------- run bar + history
function RunBar() {
  const exp = useStudio((s) => s.exp)
  const run = useResearch((s) => s.run)
  const state = useResearch()
  void state.params; void state.metrics; void state.design; void state.replicates
  const problems = specProblems(buildSpec(exp))
  const running = run?.state === 'running' || run?.state === 'queued'
  const sz = designSize(state.design, state.params, state.replicates)
  const resumable = !!run && (run.state === 'cancelled' || run.state === 'error') && JSON.stringify(buildSpec(exp)) === JSON.stringify(run.spec)
  return (
    <div className="runbar rs-runbar">
      {problems.slice(0, 3).map((p) => <div key={p} className="hint" style={{ color: 'var(--warn)' }}>{p}</div>)}
      {running
        ? <button className="btn danger" onClick={cancelStudy}><Square size={13} />Cancel study</button>
        : <button className="btn primary" disabled={problems.length > 0} onClick={startStudy}><Play size={14} />{resumable ? 'Resume study' : 'Run study'} · {sz.runs.toLocaleString()} run{sz.runs === 1 ? '' : 's'}</button>}
      {run && <div className="progress" aria-label="study progress"><i style={{ width: `${Math.round(run.fraction * 100)}%` }} /></div>}
      {run && <div className="hint">{running ? `${run.message} · ${Math.round(run.fraction * 100)} %` : run.state === 'done' ? `Done · ${run.nRuns} runs` : run.state === 'error' ? `Error: ${run.error}` : run.state === 'cancelled' ? 'Cancelled — finished runs are kept below' : run.state}</div>}
    </div>
  )
}

function History() {
  const history = useResearch((s) => s.history)
  const run = useResearch((s) => s.run)
  if (!history.length) return null
  return (
    <Sec title="Studies on this engine" extra={<HistoryIcon size={12} />}>
      <div className="rs-hist">
        {history.slice(0, 8).map((h) => (
          <button key={h.study_id} className={`rs-hist-item${run?.id === h.study_id ? ' on' : ''}`} onClick={() => void attachStudy(h.study_id)}>
            <span className={`st ${h.state}`} /><span className="nm">{h.name}</span><span className="mono">{h.n_runs}</span><small>{h.state === 'running' ? `${Math.round(h.fraction * 100)} %` : h.state}</small>
          </button>))}
      </div>
    </Sec>
  )
}

export function StudyBuilder() {
  return (
    <div className="cfg rs-cfg">
      <div className="rs-title"><Layers size={14} /><span>Study builder</span><small>over the current design</small></div>
      <Parameters />
      <DesignSection />
      <Metrics />
      <History />
      <RunBar />
    </div>
  )
}

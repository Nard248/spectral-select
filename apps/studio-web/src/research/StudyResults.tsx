import { useMemo, useState } from 'react'
import { ArrowDown, ArrowUp, FileJson, FileSpreadsheet, FileText, Layers3, Eye, SlidersHorizontal, Wand2 } from 'lucide-react'
import { WsEmpty } from '../analysis/parts'
import { downloadText, safeName } from '../analysis/exportData'
import { fmtVal } from '../ui/plots'
import { Seg, Tip } from '../ui/controls'
import type { MorrisMetric, SobolMetric, StudyRunRow } from '../api/researchTypes'
import { sobolUnreliable } from './logic'
import { MorrisChart, ParallelCoords, ScatterReps, SobolChart, C_NOISE, cmapCss } from './charts'
import {
  aggregatePoints, axisOf, fmtDuration, numOf, runsToCsv, sortRows, type SortDir,
} from './logic'
import { useResearch } from './store'
import { applyRunToDesign, loadSensitivity, loadSpecIntoBuilder, previewRun } from './actions'
const MAX_ROWS_SHOWN = 200

export function StudyResults() {
  const s = useResearch()
  const run = s.run
  if (!run) return (
    <WsEmpty icon="sweep" title="Design an experiment, then let the engine run it">
      Pick the parameters to vary, choose a design — <b style={{ display: 'inline', color: 'var(--accent)' }}>Saltelli</b> for variance-based Sobol′ indices, Morris for cheap screening, Latin hypercube or Sobol for surrogate training — and the metrics to extract.
      Runs stream in as they finish; sensitivity indices, a parallel-coordinates view and the full run table appear here.
    </WsEmpty>)
  const params = run.spec.parameters.map((p) => p.label ?? p.pointer)
  const metrics = run.spec.metrics.map((m) => m.name)
  const metric = s.metricSel && metrics.includes(s.metricSel) ? s.metricSel : metrics[0]
  const xParam = s.xParam && params.includes(s.xParam) ? s.xParam : params[0]
  const finished = run.state === 'done' || run.state === 'cancelled' || run.state === 'error'
  const nOk = s.rows.filter((r) => r.status === 'ok').length
  const nFail = s.rows.length - nOk
  const eta = run.state === 'running' && run.fraction > 0.02 ? ((Date.now() - run.startedAt) / run.fraction) * (1 - run.fraction) / 1000 : null
  const stem = safeName(`${run.spec.name}_${run.id}`)
  return (
    <>
      <div className="lf-card rs-head">
        <header>
          <Layers3 size={14} style={{ color: 'var(--accent)' }} />{run.spec.name}
          <span className={`rs-state ${run.state}`}>{run.state === 'running' ? `running · ${Math.round(run.fraction * 100)} %` : run.state}</span>
          <small>{run.spec.design.kind} · {s.rows.length}/{run.nRuns} runs · {run.spec.mode}{eta != null ? ` · ~${fmtDuration(eta)} left` : ''}{nFail ? ` · ${nFail} failed` : ''}</small>
          <span className="sp" />
          <Tip text="Open the generated Methods paragraph and provenance for this study"><button className="btn sm" onClick={() => { s.set({ tab: 'methods', methodsFor: 'study' }) }}><FileText size={13} />Methods</button></Tip>
          <Tip text="Copy this study's parameters, design and metrics back into the builder"><button className="btn sm" onClick={() => loadSpecIntoBuilder(run.spec)}><SlidersHorizontal size={13} />Reuse</button></Tip>
          <button className="btn sm" disabled={!s.rows.length} onClick={() => downloadText(`${stem}.csv`, runsToCsv(s.columns, s.rows, [`SpectralForge Studio study ${run.id} "${run.spec.name}"`, `design ${run.spec.design.kind} n=${run.spec.design.n} seed=${run.spec.design.seed}, replicates ${run.spec.replicates}, mode ${run.spec.mode}`, `p: parameter value, u: unit-cube coordinate, m: metric`]))}><FileSpreadsheet size={13} />CSV</button>
          <button className="btn sm" disabled={!s.rows.length} onClick={() => downloadText(`${stem}.json`, JSON.stringify({ study_id: run.id, spec: { ...run.spec, base_experiment: typeof run.spec.base_experiment === 'string' ? run.spec.base_experiment : '(embedded experiment)' }, manifest: run.manifest, sensitivity: s.sens, columns: s.columns, rows: s.rows }, null, 1), 'application/json')}><FileJson size={13} />JSON</button>
        </header>
        <div className="cbody">
          {run.state === 'error' && <div className="note err" style={{ margin: '0 0 8px' }}>{run.error ?? 'The study failed.'}</div>}
          {nFail > 0 && <div className="note warn" style={{ margin: '0 0 8px' }}>{nFail} of {s.rows.length} runs failed: {s.rows.find((r) => r.status !== 'ok')?.error ?? 'see the run table'}</div>}
          {run.state === 'cancelled' && <div className="note warn" style={{ margin: '0 0 8px' }}>Cancelled after {nOk} finished runs. Those results are kept and can be explored; Sobol′ indices need the complete design.</div>}
          <div className="chips2" style={{ padding: 0 }} role="tablist" aria-label="Metric">
            {metrics.map((m) => <button key={m} role="tab" aria-selected={m === metric} className={`chip2${m === metric ? ' on' : ''}`} style={{ ['--c' as string]: 'var(--accent)' }} onClick={() => s.set({ metricSel: m })}>{m}</button>)}
          </div>
        </div>
      </div>

      {!s.rows.length
        ? <div className="lf-card"><div className="cbody"><div className="hint" style={{ textAlign: 'center', padding: 24 }}><span className="spinner" style={{ width: 18, height: 18 }} /> {run.state === 'running' || run.state === 'queued' ? 'Waiting for the first run to finish…' : 'No runs were written.'}</div></div></div>
        : <>
          <SensitivityCard metric={metric} params={params} finished={finished} />
          <ResponseCard rows={s.rows} params={params} metric={metric} xParam={xParam} />
          <ParallelCard rows={s.rows} params={params} metric={metric} />
          <RunTable rows={s.rows} params={params} metrics={metrics} />
        </>}
    </>
  )
}

// ---------------------------------------------------------------------------------------------- sensitivity
function SensitivityCard({ metric, params, finished }: { metric: string; params: string[]; finished: boolean }) {
  const { run, sens, sensError, sensLoading, set } = useResearch()
  if (!run) return null
  const kind = run.spec.design.kind
  const m = sens?.metrics[metric]
  const isSobol = m?.method === 'sobol'
  return (
    <div className="lf-card">
      <header>Sensitivity <small>{kind === 'saltelli' ? 'Sobol′ indices · Saltelli 2010 / Jansen 1999 estimators, bootstrap 95 % CI' : kind === 'morris' ? 'Morris elementary effects (μ*, σ)' : 'needs a Saltelli or Morris design'}</small><span className="sp" />
        {(kind === 'saltelli' || kind === 'morris') && finished && <button className="btn sm ghost" onClick={() => void loadSensitivity()} disabled={sensLoading}>Recompute</button>}</header>
      <div className="cbody">
        {kind !== 'saltelli' && kind !== 'morris' && (
          <div className="rs-callout">
            <Wand2 size={16} />
            <div><b>Which parameter matters?</b> Variance-based sensitivity needs a Saltelli design (first-order S1 and total ST Sobol′ indices with confidence intervals, plus the Monte Carlo noise floor from replicates); Morris is a cheap screening alternative.
              <div style={{ paddingTop: 8 }}><button className="btn sm" onClick={() => { set({ design: { ...useResearch.getState().design, kind: 'saltelli', n: Math.max(16, useResearch.getState().design.n) }, params: useResearch.getState().params }); }}>Switch the builder to Saltelli</button></div></div>
          </div>)}
        {(kind === 'saltelli' || kind === 'morris') && !finished && <div className="hint" style={{ textAlign: 'center', padding: 22 }}>Indices are computed once all {run.nRuns} runs are done ({Math.round(run.fraction * 100)} %).</div>}
        {(kind === 'saltelli' || kind === 'morris') && finished && sensLoading && <div className="hint" style={{ textAlign: 'center', padding: 22 }}><span className="spinner" style={{ width: 18, height: 18 }} />Bootstrapping confidence intervals…</div>}
        {sensError && finished && <div className="note err" style={{ margin: 0 }}>{sensError}</div>}
        {m && isSobol && (
          <>
            <SobolChart metric={m as SobolMetric} params={params} />
            {sobolUnreliable(m as SobolMetric) && <div className="note warn" style={{ margin: '4px 0 0' }}>These indices are not yet reliable: ΣS1 = {(m as SobolMetric).sum_S1?.toFixed(2)} lies outside [0, 1.2] or the design has fewer than 16 base samples ({Math.round((m as SobolMetric).n_rows)} rows). Increase the base sample size n (64 or more is typical) and re-run.</div>}
            <div className="rs-facts">
              <span>Σ S1 <b>{(m as SobolMetric).sum_S1?.toFixed(2) ?? '—'}</b><Tip text="Σ S1 ≈ 1 means the model is additive; 1 − ΣS1 is the share of the variance due to interactions."><i className="info">?</i></Tip></span>
              <span>variance <b>{fmtVal((m as SobolMetric).variance ?? NaN)}</b></span>
              <span style={{ color: C_NOISE }}>MC noise floor <b>{(m as SobolMetric).noise.noise_floor != null ? `${((m as SobolMetric).noise.noise_floor! * 100).toFixed(2)} %` : 'n/a (1 replicate)'}</b></span>
              <span>{(m as SobolMetric).n_rows} rows</span>
            </div>
          </>)}
        {m && !isSobol && <MorrisChart metric={m as MorrisMetric} params={params} />}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------------------------- response scatter
function ResponseCard({ rows, params, metric, xParam }: { rows: StudyRunRow[]; params: string[]; metric: string; xParam: string }) {
  const { run, selectedRun, set, brush } = useResearch()
  const [colorBy, setColorBy] = useState<string>('')
  const pts = useMemo(() => aggregatePoints(rows, xParam, metric), [rows, xParam, metric])
  const spec = run?.spec.parameters.find((p) => (p.label ?? p.pointer) === xParam)
  const selPoint = rows.find((r) => r.run_id === selectedRun)?.point ?? null
  const cAxis = useMemo(() => (colorBy ? axisOf(rows, `p:${colorBy}`, colorBy, run?.spec.parameters.find((p) => (p.label ?? p.pointer) === colorBy)?.scale === 'log') : null), [rows, colorBy, run])
  const color = (p: (typeof pts)[number]) => { if (!cAxis) return '#35d8ff'; const v = numOf(p.rows[0], cAxis.key); const t = cAxis.log ? (Math.log(v) - Math.log(cAxis.lo)) / (Math.log(cAxis.hi) - Math.log(cAxis.lo)) : (v - cAxis.lo) / (cAxis.hi - cAxis.lo); return cmapCss('viridis', t) }
  const filtered = Object.keys(brush).length > 0
  return (
    <div className="lf-card">
      <header>{metric} <small>against {xParam} · {pts.length} design point{pts.length === 1 ? '' : 's'}{pts.some((p) => p.n > 1) ? ' · error bars: ±1 sd over replicates' : ''}</small><span className="sp" />
        {params.length > 1 && <span className="rs-hdr-sel"><span className="hint">colour</span><select className="field" value={colorBy} aria-label="Colour points by" onChange={(e) => setColorBy(e.target.value)}><option value="">—</option>{params.filter((p) => p !== xParam).map((p) => <option key={p} value={p}>{p}</option>)}</select></span>}</header>
      <div className="cbody">
        {params.length > 1 && <div className="chips2" style={{ padding: '0 0 8px' }} role="tablist" aria-label="x parameter">
          {params.map((p) => <button key={p} role="tab" aria-selected={p === xParam} className={`chip2${p === xParam ? ' on' : ''}`} style={{ ['--c' as string]: 'var(--accent-2)' }} onClick={() => set({ xParam: p })}>{p}</button>)}
        </div>}
        <ScatterReps points={pts} metricKey={`m:${metric}`} xLabel={`${xParam}${spec ? '' : ''}`} yLabel={metric} logX={spec?.scale === 'log'} color={color} selectedPoint={selPoint}
          onSelect={(pt) => { const r = rows.find((x) => x.point === pt); if (r) set({ selectedRun: r.run_id }) }} />
        {params.length > 1 && <div className="hint" style={{ paddingTop: 4 }}>Each dot is one design point; the other parameters vary between dots, so scatter around the dashed trend is the effect of everything else{cAxis ? ` (colour = ${colorBy})` : ''}.</div>}
        {filtered && <div className="hint" style={{ color: 'var(--accent)' }}>A parallel-coordinates brush is active ({Object.keys(brush).length} axis).</div>}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------------------------- parallel coordinates
function ParallelCard({ rows, params, metric }: { rows: StudyRunRow[]; params: string[]; metric: string }) {
  const { run, brush, selectedRun, set } = useResearch()
  const axes = useMemo(() => [
    ...params.map((p) => axisOf(rows, `p:${p}`, p, run?.spec.parameters.find((q) => (q.label ?? q.pointer) === p)?.scale === 'log')),
    axisOf(rows, `m:${metric}`, metric),
  ], [rows, params, metric, run])
  if (params.length < 1) return null
  return (
    <div className="lf-card">
      <header>Parallel coordinates <small>every run is a line · coloured by {metric}</small><span className="sp" /><span className="hint">drag an axis to brush · double-click clears</span></header>
      <div className="cbody"><ParallelCoords rows={rows} axes={axes} colorKey={`m:${metric}`} brush={brush} onBrush={(b) => set({ brush: b })} selected={selectedRun} onSelect={(id) => set({ selectedRun: id })} /></div>
    </div>
  )
}

// ---------------------------------------------------------------------------------------------- runs table
function RunTable({ rows, params, metrics }: { rows: StudyRunRow[]; params: string[]; metrics: string[] }) {
  const { run, brush, selectedRun, set } = useResearch()
  const [sort, setSort] = useState<{ col: string; dir: SortDir }>({ col: 'run_id', dir: 'asc' })
  const [showAll, setShowAll] = useState(false)
  const [okOnly, setOkOnly] = useState(false)
  const axes = useMemo(() => params.map((p) => axisOf(rows, `p:${p}`, p)).concat(metrics.map((m) => axisOf(rows, `m:${m}`, m))), [rows, params, metrics])
  const filtered = useMemo(() => rows.filter((r) => (!okOnly || r.status === 'ok') && axes.every((a) => { const b = brush[a.key]; if (!b) return true; const t = (numOf(r, a.key) - a.lo) / (a.hi - a.lo); return t >= b[0] - 1e-9 && t <= b[1] + 1e-9 })), [rows, brush, axes, okOnly])
  const sorted = useMemo(() => sortRows(filtered, sort.col, sort.dir), [filtered, sort])
  const shown = showAll ? sorted : sorted.slice(0, MAX_ROWS_SHOWN)
  const cols = [{ k: 'run_id', l: 'run' }, { k: 'replicate', l: 'rep' }, ...params.map((p) => ({ k: `p:${p}`, l: p })), ...metrics.map((m) => ({ k: `m:${m}`, l: m })), { k: 't_capture_s', l: 't (s)' }]
  const sel = rows.find((r) => r.run_id === selectedRun) ?? null
  const th = (c: { k: string; l: string }) => (
    <th key={c.k} onClick={() => setSort({ col: c.k, dir: sort.col === c.k && sort.dir === 'asc' ? 'desc' : 'asc' })} aria-sort={sort.col === c.k ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none'} className={c.k.startsWith('m:') ? 'm' : ''} title={c.l}>
      <span>{c.l.length > 16 ? `${c.l.slice(0, 15)}…` : c.l}{sort.col === c.k && (sort.dir === 'asc' ? <ArrowUp size={10} /> : <ArrowDown size={10} />)}</span></th>)
  return (
    <>
      {sel && run && (
        <div className="lf-card rs-sel">
          <header><Eye size={14} style={{ color: 'var(--accent)' }} />Run {sel.run_id}<small>point {sel.point} · replicate {sel.replicate} · seed {sel.seed}{sel.exp_hash ? ` · hash ${String(sel.exp_hash).slice(0, 10)}` : ''}</small><span className="sp" />
            <Tip text="Apply this run's parameter values to the experiment in Design (one undo step)"><button className="btn sm" onClick={() => applyRunToDesign(sel, run.spec)}>Apply to Design</button></Tip>
            <Tip text="Apply the values and take an engine preview of this exact point"><button className="btn sm primary" onClick={() => void previewRun(sel, run.spec)}>Preview this run</button></Tip>
            <button className="btn sm ghost" onClick={() => set({ selectedRun: null })}>Close</button></header>
          <div className="cbody rs-kv">
            {params.map((p) => <div key={p}><span>{p}</span><b>{fmtVal(numOf(sel, `p:${p}`))}</b></div>)}
            {metrics.map((m) => <div key={m} className="m"><span>{m}</span><b>{fmtVal(numOf(sel, `m:${m}`))}</b></div>)}
            {sel.status !== 'ok' && <div className="err" style={{ gridColumn: '1/-1' }}>{String(sel.error ?? 'failed')}</div>}
            {run.spec.mode && <div className="hint" style={{ gridColumn: '1/-1' }}>{useResearch.getState().keepCaptures ? 'This capture was kept on the engine (captures/<run_id>.npz in the study folder; reload it with spectraforge.studio.api.load_capture). ' : ''}Captures are not streamed to the browser: use “Preview this run” to see the point in the Explorer.</div>}
          </div>
        </div>)}
      <div className="lf-card">
        <header>Runs <small>{filtered.length === rows.length ? `${rows.length}` : `${filtered.length} of ${rows.length} (brushed)`}</small><span className="sp" />
          <Seg small value={okOnly ? 'ok' : 'all'} onChange={(v) => setOkOnly(v === 'ok')} options={[{ value: 'all', label: 'All' }, { value: 'ok', label: 'Succeeded' }]} /></header>
        <div className="cbody" style={{ padding: 0 }}>
          <div className="rs-table-wrap">
            <table className="stat rs-table"><thead><tr>{cols.map(th)}</tr></thead>
              <tbody>{shown.map((r) => (
                <tr key={r.run_id} className={`${r.run_id === selectedRun ? 'sel' : ''}${r.status !== 'ok' ? ' fail' : ''}`} tabIndex={0} onClick={() => set({ selectedRun: r.run_id })} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); set({ selectedRun: r.run_id }) } }}>
                  {cols.map((c) => <td key={c.k} className={c.k.startsWith('m:') ? 'm' : ''}>{c.k === 'run_id' ? <span style={{ color: r.status === 'ok' ? 'var(--text)' : 'var(--err)' }}>{String(r.run_id)}</span> : c.k === 'replicate' ? String(r.replicate) : typeof r[c.k] === 'number' ? fmtVal(r[c.k] as number) : '—'}</td>)}
                </tr>))}</tbody></table>
          </div>
          {sorted.length > MAX_ROWS_SHOWN && <div style={{ padding: 8, textAlign: 'center' }}><button className="btn sm ghost" onClick={() => setShowAll(!showAll)}>{showAll ? `Show the first ${MAX_ROWS_SHOWN}` : `Show all ${sorted.length}`}</button></div>}
        </div>
      </div>
    </>
  )
}

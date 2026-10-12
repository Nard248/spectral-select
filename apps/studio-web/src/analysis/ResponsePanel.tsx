import { useEffect, useMemo } from 'react'
import { FileJson, FileSpreadsheet } from 'lucide-react'
import type { CaptureMeta, ResponseRequest } from '../api/types'
import { useAnalysis } from './store'
import { isRegion, type NamedRoi } from './types'
import { useResponse } from './hooks'
import { XYChart, fmtVal, type XYSeries } from '../ui/plots'
import { Num, Sel, Toggle } from '../ui/controls'
import { normSpecClean } from './norm'
import { defaultGt, gtLabel, gtUnitOf, numericGt } from './gt'
import { MetricPicker, defaultMetric } from './MetricPicker'
import { downloadText, responseToCsv, safeName, toJson } from './exportData'
import { metricLabel, metricUnit } from './metrics'
import { Busy, ChartBox, WsEmpty } from './parts'

function pearson(x: number[], y: number[]): number {
  const n = Math.min(x.length, y.length)
  if (n < 3) return NaN
  let sx = 0, sy = 0
  for (let i = 0; i < n; i++) { sx += x[i]; sy += y[i] }
  const mx = sx / n, my = sy / n
  let a = 0, b = 0, c = 0
  for (let i = 0; i < n; i++) { const dx = x[i] - mx, dy = y[i] - my; a += dx * dy; b += dx * dx; c += dy * dy }
  return a / Math.sqrt(b * c || 1)
}

/** The headline plot: a per-pixel metric versus a ground-truth variable (e.g. fluorescence versus cover thickness). */
export function ResponsePanel({ meta, ex }: { meta: CaptureMeta; ex: number | null }) {
  const rois = useAnalysis((s) => s.rois)
  const norm = useAnalysis((s) => s.norm)
  const st = useAnalysis((s) => s.response)
  const setResp = useAnalysis((s) => s.setResponse)
  const gts = useMemo(() => numericGt(meta), [meta])
  const regions = useMemo(() => rois.filter((r): r is NamedRoi => isRegion(r.spec)), [rois])

  const gt = st.gt && gts.includes(st.gt) ? st.gt : defaultGt(meta)
  const metricOk = st.metric && (st.metric.kind === 'band_ratio' ? meta.excitations_nm.includes(st.metric.a.ex) && meta.excitations_nm.includes(st.metric.b.ex) : meta.excitations_nm.includes(st.metric.ex))
  useEffect(() => { if (!metricOk) setResp({ metric: defaultMetric(meta, ex) }) }, [metricOk, meta, ex, setResp])
  const metric = metricOk ? st.metric! : defaultMetric(meta, ex)
  const roi = regions.find((r) => r.id === st.roiId) ?? null

  const req = useMemo<ResponseRequest | null>(() => (gt ? { x_ground_truth: gt, metric, roi: roi ? (roi.spec as ResponseRequest['roi']) : null, bins: st.bins, norm: normSpecClean(norm) } : null), [gt, metric, roi, st.bins, norm])
  const resp = useResponse(meta.id, req)
  const R = resp.data
  const unit = R ? metricUnit(metric, R.unit) : ''

  const series = useMemo<XYSeries[]>(() => R ? [{ name: 'binned mean ± std', x: R.bin_centres, y: R.mean, std: R.std, color: '#35d8ff', width: 2.2, points: true }] : [], [R])
  const scatter = useMemo(() => (R && st.scatter ? [{ x: R.scatter_x, y: R.scatter_y, color: '#35d8ff', r: 1.3, opacity: 0.16 }] : []), [R, st.scatter])
  const r = R ? pearson(R.scatter_x, R.scatter_y) : NaN
  const first = R?.mean[0], last = R ? R.mean[R.mean.length - 1] : undefined

  if (!gts.length) return <WsEmpty icon="response" title="No numeric ground truth">This capture carries no numeric ground-truth maps, so there is nothing to plot a response against.</WsEmpty>

  const ctx = { captureId: meta.id, experimentName: meta.name, norm: normSpecClean(norm), roiName: roi?.name ?? 'all pixels', roi: roi?.spec ?? null, extra: { metric: metricLabel(metric) } }
  const stem = safeName(`${meta.name}_${gt}_response`)
  return (
    <div className="scroll">
      <div className="asec" style={{ paddingTop: 0 }}>
        <div className="h" style={{ paddingTop: 10 }}>Signal versus <small>ground truth on the x axis</small><span className="sp" /><Busy on={resp.loading} /></div>
        <div className="ctrls" style={{ paddingTop: 2 }}>
          <div style={{ flex: 1, minWidth: 160 }}><Sel value={gt ?? ''} options={gts.map((g) => ({ value: g, label: `${gtLabel(g)}${gtUnitOf(g) ? ` (${gtUnitOf(g)})` : ''}` }))} onChange={(g) => setResp({ gt: g })} /></div>
        </div>
        <div className="pad" style={{ paddingTop: 2 }}><div className="hint" style={{ padding: '0 2px 2px' }}>Metric (y axis)</div></div>
        <MetricPicker meta={meta} value={metric} onChange={(m) => setResp({ metric: m })} />
        <div className="ctrls">
          <span className="grp">ROI</span>
          <div style={{ minWidth: 130, flex: 1 }}><Sel value={st.roiId ?? ''} options={[{ value: '', label: 'all pixels' }, ...regions.map((q) => ({ value: q.id, label: q.name }))]} onChange={(v) => setResp({ roiId: v || null })} /></div>
          <span className="grp">bins</span><div style={{ width: 62 }}><Num value={st.bins} min={4} max={64} step={1} integer bar={false} onChange={(bins) => setResp({ bins })} /></div>
          <Toggle value={st.scatter} label="scatter" onChange={(scatter) => setResp({ scatter })} />
        </div>
      </div>

      <div className="asec">
        <div className="h">{metricLabel(metric)} <small>vs {gt ? gtLabel(gt) : ''}</small></div>
        <div className="pad">
          {resp.error && <div className="note err">{resp.error}</div>}
          <ChartBox><XYChart series={series} scatter={scatter} height={270} zeroBase={metric.kind !== 'peak_wavelength' && metric.kind !== 'band_ratio'} xLabel={gt ? `${gtLabel(gt)}${gtUnitOf(gt) ? ` (${gtUnitOf(gt)})` : ''}` : ''} yLabel={unit}
            tipTitle={(x) => `${gt ? gtLabel(gt) : 'x'} ≈ ${fmtVal(x)} ${gt ? gtUnitOf(gt) : ''}`} legend /></ChartBox>
        </div>
        {R && R.bin_centres.length > 0 && <dl className="kv" style={{ paddingTop: 8 }}>
          <dt>pixels</dt><dd>{R.count.reduce((a, b) => a + b, 0).toLocaleString()} in {R.bin_centres.length} bins{roi ? ` · ${roi.name}` : ''}</dd>
          <dt>correlation r</dt><dd>{Number.isFinite(r) ? r.toFixed(3) : '—'}</dd>
          {first != null && last != null && Number.isFinite(first) && Number.isFinite(last) && first !== 0 && metric.kind !== 'peak_wavelength' && <><dt>first → last bin</dt><dd>{fmtVal(first)} → {fmtVal(last)} {unit} ({((last / first - 1) * 100).toFixed(0)} %)</dd></>}
          {first != null && last != null && metric.kind === 'peak_wavelength' && <><dt>peak shift</dt><dd>{(last - first).toFixed(1)} nm</dd></>}
        </dl>}
      </div>

      {R && R.bin_centres.length > 0 && <div className="asec">
        <div className="h">Binned values <small>{R.bin_centres.length} bins</small></div>
        <div className="tscroll"><table className="stat"><thead><tr><th>{gt ? gtLabel(gt) : 'x'}</th><th>mean</th><th>± std</th><th>n</th></tr></thead>
          <tbody>{R.bin_centres.map((x, i) => <tr key={i}><td>{fmtVal(x)}</td><td>{fmtVal(R.mean[i])}</td><td>{fmtVal(R.std[i])}</td><td>{R.count[i]}</td></tr>)}</tbody></table></div>
      </div>}
      <div className="exports">
        <span className="hint">Export response</span>
        <button className="btn sm" disabled={!R} onClick={() => R && downloadText(`${stem}.csv`, responseToCsv(R, ctx))}><FileSpreadsheet size={13} />CSV</button>
        <button className="btn sm" disabled={!R} onClick={() => R && downloadText(`${stem}.json`, toJson('response', ctx, R), 'application/json')}><FileJson size={13} />JSON</button>
      </div>
      <div style={{ height: 14 }} />
    </div>
  )
}

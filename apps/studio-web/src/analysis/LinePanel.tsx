import { useEffect, useMemo } from 'react'
import { FileJson, FileSpreadsheet, X } from 'lucide-react'
import type { CaptureMeta, LineRequest } from '../api/types'
import { useAnalysis } from './store'
import { isLine, type LineSpec, type NamedRoi } from './types'
import { useLine } from './hooks'
import { XYChart, fmtVal, type XYSeries } from '../ui/plots'
import { Num, Sel, Toggle } from '../ui/controls'
import { wavelengthToHex } from '../lib/color'
import { normSpecClean } from './norm'
import { defaultGt, gtLabel, gtUnitOf, numericGt } from './gt'
import { downloadText, lineToCsv, safeName, toJson } from './exportData'
import { lineLength } from './roi'
import { Busy, ChartBox, WsEmpty } from './parts'

export function LinePanel({ meta, ex }: { meta: CaptureMeta; ex: number | null }) {
  const rois = useAnalysis((s) => s.rois)
  const selectedId = useAnalysis((s) => s.selectedId)
  const select = useAnalysis((s) => s.select)
  const setSpec = useAnalysis((s) => s.setSpec)
  const norm = useAnalysis((s) => s.norm)
  const line = useAnalysis((s) => s.line)
  const setLine = useAnalysis((s) => s.setLine)
  const lines = useMemo(() => rois.filter((r): r is NamedRoi & { spec: LineSpec } => isLine(r.spec)), [rois])
  const cur = lines.find((r) => r.id === selectedId) ?? lines[0] ?? null
  const gts = useMemo(() => numericGt(meta), [meta])

  const exs = meta.excitations_nm
  const lineEx = line.exNm != null && exs.includes(line.exNm) ? line.exNm : (ex != null && exs.includes(ex) ? ex : exs[0])
  const emission = useMemo(() => meta.emission_nm[String(lineEx)] ?? [], [meta, lineEx])
  // default bands: three evenly spaced, re-chosen whenever the excitation changes the band set
  useEffect(() => {
    if (!emission.length) return
    const ok = line.bandsNm.length > 0 && line.bandsNm.every((b) => emission.some((e) => Math.abs(e - b) < 0.5))
    if (!ok) {
      const st = meta.stats[String(lineEx)] ?? []
      let bi = 0
      st.forEach((q, i) => { if (q.p99 > (st[bi]?.p99 ?? -Infinity)) bi = i })
      const near = (nm: number) => emission.reduce((b, e) => (Math.abs(e - nm) < Math.abs(b - nm) ? e : b), emission[0])
      const pk = emission[bi] ?? emission[0]
      setLine({ bandsNm: [...new Set([pk, near(pk + 30), near(pk + 70)])].sort((a, b) => a - b) })
    }
  }, [emission, line.bandsNm, setLine, meta.stats, lineEx])
  const gt = line.gt && gts.includes(line.gt) ? line.gt : defaultGt(meta)

  const req = useMemo<LineRequest | null>(() => (cur ? { p0: cur.spec.p0, p1: cur.spec.p1, width_px: cur.spec.width_px, norm: normSpecClean(norm), excitations_nm: [lineEx], ground_truth: gt ? [gt] : [] } : null), [cur, norm, lineEx, gt])
  const prof = useLine(meta.id, req)
  const P = prof.data
  const useMm = !!(P?.distance_mm && line.mmAxis)

  const series = useMemo<XYSeries[]>(() => {
    if (!P) return []
    const xs = useMm ? P.distance_mm! : P.distance_px
    const em = P.emission_nm[String(lineEx)] ?? []
    const rows = P.value[String(lineEx)] ?? []
    const out: XYSeries[] = line.bandsNm.flatMap((nm) => {
      const bi = em.reduce((b, e, i) => (Math.abs(e - nm) < Math.abs(em[b] - nm) ? i : b), 0)
      return [{ name: `${em[bi]?.toFixed(0)} nm`, x: xs, y: rows.map((r) => r[bi]), color: wavelengthToHex(em[bi] ?? nm), width: 2 }]
    })
    if (gt && P.ground_truth[gt]) out.push({ name: gtLabel(gt), x: xs, y: P.ground_truth[gt], color: '#f2f5fa', dash: '5 4', width: 1.5, axis: 'r' })
    return out
  }, [P, line.bandsNm, lineEx, gt, useMm])

  if (!cur) return (
    <WsEmpty icon="line" title="Draw a line across the sample">
      Pick the <b style={{ display: 'inline', color: 'var(--accent)' }}>line</b> tool (<span className="kbd">L</span>) and drag across the image — e.g. along the wedge.
      You get the bands of a chosen excitation versus distance, with a ground-truth trace such as cover thickness on a second axis.
    </WsEmpty>)

  const ctx = () => ({ captureId: meta.id, experimentName: meta.name, norm: normSpecClean(norm), roiName: cur.name, roi: cur.spec })
  const stem = safeName(`${meta.name}_${cur.name}_profile`)
  const gtVals = gt && P?.ground_truth[gt] ? P.ground_truth[gt].filter(Number.isFinite) : []
  const unit = P?.unit ?? ''
  const addable = emission.filter((e) => !line.bandsNm.some((b) => Math.abs(b - e) < 0.5))

  return (
    <div className="scroll">
      {lines.length > 1 && <div className="chips2">{lines.map((r) => <button key={r.id} className={`chip2${r.id === cur.id ? ' on' : ''}`} style={{ ['--c' as string]: r.color }} onClick={() => select(r.id)}><i />{r.name}</button>)}</div>}
      <div className="ctrls">
        <span className="grp">excitation</span>
        <div className="tabs" style={{ flex: 'none' }}>{exs.map((e) => <button key={e} className={`tab${e === lineEx ? ' on' : ''}`} style={{ ['--c' as string]: wavelengthToHex(e) }} onClick={() => setLine({ exNm: e })}><i />{e}</button>)}</div>
      </div>
      <div className="ctrls" style={{ paddingTop: 0 }}>
        <span className="grp">width</span><div style={{ width: 76 }}><Num value={cur.spec.width_px} min={1} max={25} step={1} integer unit="px" bar={false} onChange={(v) => setSpec(cur.id, { ...cur.spec, width_px: v })} /></div>
        <span className="grp" style={{ marginLeft: 6 }}>ground truth</span>
        <div style={{ flex: 1, minWidth: 130 }}><Sel value={gt ?? ''} options={[{ value: '', label: '— none —' }, ...gts.map((g) => ({ value: g, label: gtLabel(g) }))]} onChange={(v) => setLine({ gt: v || null })} /></div>
        {P?.distance_mm && <Toggle value={line.mmAxis} label="mm" onChange={(mmAxis) => setLine({ mmAxis })} tip="Distance in millimetres along the sample surface (when depth is known) instead of pixels." />}
      </div>
      <div className="chips2">
        {line.bandsNm.map((b) => <span key={b} className="chip2 on mono" style={{ ['--c' as string]: wavelengthToHex(b) }}><i />{b.toFixed(0)} nm<button style={{ border: 0, background: 'transparent', color: 'var(--faint)', display: 'inline-flex', padding: 0, cursor: 'pointer' }} aria-label={`remove band ${b}`} onClick={() => setLine({ bandsNm: line.bandsNm.filter((q) => q !== b) })}><X size={11} /></button></span>)}
        {line.bandsNm.length < 6 && addable.length > 0 && <div style={{ width: 110 }}><Sel value={'' as unknown as number} options={[{ value: '' as unknown as number, label: '+ add band' }, ...addable.map((e) => ({ value: e, label: `${e.toFixed(0)} nm` }))]} onChange={(v) => { if (v !== ('' as unknown as number)) setLine({ bandsNm: [...line.bandsNm, v].sort((a, b) => a - b) }) }} /></div>}
      </div>

      <div className="asec">
        <div className="h">Profile <small>{cur.name} · {lineLength(cur.spec).toFixed(0)} px{P?.distance_mm ? ` · ${P.distance_mm[P.distance_mm.length - 1].toFixed(1)} mm` : ''} · {lineEx} nm excitation</small><span className="sp" /><Busy on={prof.loading} /></div>
        <div className="pad">
          {prof.error && <div className="note err">{prof.error}</div>}
          <ChartBox><XYChart series={series} height={230} xLabel={useMm ? 'distance (mm)' : 'distance (px)'} yLabel={unit} yLabelR={gt ? `${gtLabel(gt)}${gtUnitOf(gt) ? ` (${gtUnitOf(gt)})` : ''}` : undefined} legend
            tipTitle={(x) => `${useMm ? 'd' : 'd'} = ${fmtVal(x)} ${useMm ? 'mm' : 'px'}`} /></ChartBox>
        </div>
        {gtVals.length > 0 && <dl className="kv" style={{ paddingTop: 8 }}>
          <dt>{gtLabel(gt!)} along line</dt><dd>{fmtVal(Math.min(...gtVals))} → {fmtVal(Math.max(...gtVals))} {gtUnitOf(gt!)}</dd>
          <dt>samples</dt><dd>{P?.distance_px.length} · width {cur.spec.width_px} px</dd>
        </dl>}
      </div>
      <div className="exports">
        <span className="hint">Export profile</span>
        <button className="btn sm" disabled={!P} onClick={() => P && downloadText(`${stem}.csv`, lineToCsv(P, { ...ctx(), extra: { excitation_nm: lineEx, width_px: cur.spec.width_px } }))}><FileSpreadsheet size={13} />CSV</button>
        <button className="btn sm" disabled={!P} onClick={() => P && downloadText(`${stem}.json`, toJson('line_profile', ctx(), P), 'application/json')}><FileJson size={13} />JSON</button>
      </div>
      <div style={{ height: 14 }} />
    </div>
  )
}

import type { CaptureMeta, MetricSpec } from '../api/types'
import { Sel, Seg } from '../ui/controls'
import { brightestOf } from './gt'
import { LEAK_HINT, isLayeredCapture, readsLeakBand } from './layeredDefaults'

export function defaultMetric(meta: CaptureMeta, ex?: number | null): MetricSpec {
  const b = brightestOf(meta, ex)
  return { kind: 'band', ex: b.ex, band_nm: b.band_nm }
}

const nearest = (arr: number[], v: number) => arr.reduce((b, e) => (Math.abs(e - v) < Math.abs(b - v) ? e : b), arr[0])

/** Choose a per-pixel scalar: single band, band ratio, band-range sum, or peak wavelength. */
export function MetricPicker({ meta, value, onChange }: { meta: CaptureMeta; value: MetricSpec; onChange: (m: MetricSpec) => void }) {
  const exs = meta.excitations_nm
  const em = (ex: number) => meta.emission_nm[String(ex)] ?? []
  const exSel = (v: number, set: (e: number) => void) => <Sel value={v} options={exs.map((e) => ({ value: e, label: `ex ${e} nm` }))} onChange={set} />
  const bandSel = (ex: number, v: number, set: (b: number) => void) => <Sel value={nearest(em(ex), v)} options={em(ex).map((e) => ({ value: e, label: `${e.toFixed(0)} nm` }))} onChange={set} />
  const sw = (kind: MetricSpec['kind']) => {
    const d = defaultMetric(meta, value.kind === 'band_ratio' ? value.a.ex : 'ex' in value ? value.ex : undefined)
    const ex = d.kind === 'band' ? d.ex : exs[0], b = d.kind === 'band' ? d.band_nm : em(ex)[0]
    const bs = em(ex)
    if (kind === 'band') return onChange({ kind, ex, band_nm: b })
    if (kind === 'band_ratio') return onChange({ kind, a: { ex, band_nm: b }, b: { ex, band_nm: bs[Math.max(0, bs.length - 3)] } })
    if (kind === 'band_sum') return onChange({ kind, ex, from_nm: bs[0], to_nm: bs[bs.length - 1] })
    return onChange({ kind, ex })
  }
  return (
    <div className="metric-box">
      <Seg small value={value.kind} onChange={sw} options={[{ value: 'band', label: 'Band' }, { value: 'band_ratio', label: 'Ratio' }, { value: 'band_sum', label: 'Sum' }, { value: 'peak_wavelength', label: 'Peak λ' }]} />
      {value.kind === 'band' && <div className="two">{exSel(value.ex, (ex) => onChange({ ...value, ex, band_nm: nearest(em(ex), value.band_nm) }))}{bandSel(value.ex, value.band_nm, (band_nm) => onChange({ ...value, band_nm }))}</div>}
      {value.kind === 'band_ratio' && <>
        <div className="row2"><label>A</label><div className="two">{exSel(value.a.ex, (ex) => onChange({ ...value, a: { ex, band_nm: nearest(em(ex), value.a.band_nm) } }))}{bandSel(value.a.ex, value.a.band_nm, (band_nm) => onChange({ ...value, a: { ...value.a, band_nm } }))}</div></div>
        <div className="row2"><label>B</label><div className="two">{exSel(value.b.ex, (ex) => onChange({ ...value, b: { ex, band_nm: nearest(em(ex), value.b.band_nm) } }))}{bandSel(value.b.ex, value.b.band_nm, (band_nm) => onChange({ ...value, b: { ...value.b, band_nm } }))}</div></div></>}
      {value.kind === 'band_sum' && <>
        {exSel(value.ex, (ex) => onChange({ ...value, ex, from_nm: em(ex)[0], to_nm: em(ex)[em(ex).length - 1] }))}
        <div className="two">{bandSel(value.ex, value.from_nm ?? em(value.ex)[0], (from_nm) => onChange({ ...value, from_nm }))}{bandSel(value.ex, value.to_nm ?? em(value.ex)[em(value.ex).length - 1], (to_nm) => onChange({ ...value, to_nm }))}</div></>}
      {value.kind === 'peak_wavelength' && exSel(value.ex, (ex) => onChange({ ...value, ex }))}
      {isLayeredCapture(meta) && <div className={`hint leakhint${readsLeakBand(value) ? ' warn' : ''}`} style={{ padding: '2px 2px 0', color: readsLeakBand(value) ? 'var(--warn, #f2b84b)' : undefined }}>
        {readsLeakBand(value) ? '⚠ ' : 'ⓘ '}{LEAK_HINT.charAt(0).toUpperCase() + LEAK_HINT.slice(1)} — use a band ≈ ex + 40 nm.</div>}
    </div>
  )
}

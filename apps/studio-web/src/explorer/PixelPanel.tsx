import { fmtSig, gtUnit } from '../lib/format'
import { Fragment, useMemo, useState } from 'react'
import { usePixel } from './data'
import { LineChart, StackedArea, Heatmap } from '../ui/charts'
import { Empty, Tip } from '../ui/controls'
import { wavelengthToHex } from '../lib/color'
import { fluorophoreMap } from '../state/presets'
import { useStudio } from '../state/store'
import { dyeColor } from '../panels/inspector/MaterialInspector'
import type { CaptureMeta } from '../api/types'

const fmtSci = (v: number) => fmtSig(v)
const COMP_LABELS: Record<string, string> = { elastic: 'elastic scatter', glint: 'specular glint', raman: 'water Raman', leak_source: 'excitation leak', room_light: 'room light' }
const compLabel = (n: string) => (n.startsWith('background:') ? `background (${n.slice(11).replace(/_/g, ' ')})` : COMP_LABELS[n] ?? n.replace(/_/g, ' '))
const COMP_COLORS: Record<string, string> = { elastic: '#d6dbe3', glint: '#ffffff', raman: '#5ec8ff', leak_source: '#ff6b5e', room_light: '#ffd35e' }
const PALETTE = ['#4ef28a', '#ff9b4a', '#b48cff', '#35d8ff', '#ff6bb0', '#e8e35a', '#6b9dff', '#ff6b5e']

/** Resample per-excitation emission grids onto one common grid (union of wavelengths). */
function toEem(emission: Record<string, number[]>, values: Record<string, number[]>) {
  const exs = Object.keys(values).map(Number).sort((a, b) => a - b)
  const grid = [...new Set(Object.values(emission).flat())].sort((a, b) => a - b)
  const data = exs.map((ex) => {
    const em = emission[String(ex)], v = values[String(ex)]
    return grid.map((g) => { const j = em.indexOf(g); return j >= 0 ? v[j] : 0 })
  })
  return { exs, grid, data }
}

export function PixelPanel({ meta, pixel, ex }: { meta: CaptureMeta; pixel: { x: number; y: number } | null; ex: number }) {
  const { data: px, loading, error } = usePixel(meta.id, pixel?.x ?? null, pixel?.y ?? null)
  const exp = useStudio((s) => s.exp)
  const fl = useMemo(() => fluorophoreMap(exp), [exp])
  const exKey = String(ex)
  const [allGt, setAllGt] = useState(false)

  const layers = useMemo(() => {
    if (!px?.components[exKey]) return []
    let pi = 0
    return Object.entries(px.components[exKey]).map(([name, y]) => {
      const nameBase = name.split('@')[0], layer = name.includes('@') ? name.split('@')[1] : null
      const key = nameBase.replace(/^(dye|secondary|background):/, '')
      const dye = fl[key]
      const base = nameBase.split(':')[0]
      const suffix = layer ? ` · ${layer}` : ''
      const color = nameBase.startsWith('dye:') ? dyeColor(dye) : nameBase.startsWith('secondary:') ? dyeColor(dye) : COMP_COLORS[base] ?? PALETTE[pi++ % PALETTE.length]
      return { name: (name.startsWith('secondary:') ? `${key} (re-emitted)` : name.startsWith('dye:') ? key : compLabel(nameBase)) + suffix, color, y: Array.from(y as ArrayLike<number>) }
    }).filter((l) => l.y.some((v) => v > 1e-9))
  }, [px, exKey, fl])

  const contribTotal = useMemo(() => layers.reduce((a, l) => a + l.y.reduce((s, v) => s + Math.max(0, v), 0), 0), [layers])
  const contrib = useMemo(() => layers.map((l) => { const sum = l.y.reduce((s, v) => s + Math.max(0, v), 0); return { name: l.name, color: l.color, sum, pct: contribTotal > 0 ? (100 * sum) / contribTotal : 0 } })
    .sort((a, b) => b.sum - a.sum), [layers, contribTotal])

  const specSeries = useMemo(() => px ? Object.keys(px.adu).map((k) => ({ name: `${k} nm`, x: px.emission_nm[k], y: px.adu[k], color: wavelengthToHex(Number(k)), width: k === exKey ? 2.2 : 1.1, opacity: k === exKey ? 1 : 0.55 })) : [], [px, exKey])
  const eemd = useMemo(() => (px ? toEem(px.emission_nm, px.radiance) : null), [px])

  return (
    <div className="scroll">
      {!pixel && <Empty title="Click a pixel">The image is a view into the 4D cube (excitation × emission × y × x). Click anywhere to decompose that pixel's spectrum into physical components — or draw regions with the ROI tools for EEMs, profiles and response curves.</Empty>}
      {pixel && <>
        <div className="plot-title">Pixel ({pixel.x}, {pixel.y}) {loading && <small>loading…</small>}{error && <small style={{ color: 'var(--err)' }}>{error}</small>}</div>
        {px && <>
          <div className="plot-title" style={{ paddingTop: 2 }}>Component decomposition <small>at {ex} nm excitation</small></div>
          <div style={{ padding: '0 8px' }}>
            <Tip text="Noise-free radiance split into physical origin: each dye's fluorescence, elastic leak of excitation light, water Raman, glints, room light. The dashed line is the engine's total radiance. This is the unsupervised-unmixing ground truth.">
              <div><StackedArea x={px.emission_nm[exKey]} layers={layers} height={170} xLabel="emission nm" yLabel="radiance" yUnit="mW mm⁻² sr⁻¹ / band" scaleBar
                overlay={{ y: px.radiance[exKey], color: '#ffffff', name: 'total radiance' }} /></div></Tip>
          </div>
          {contrib.length > 0 && <div className="contrib">
            <div className="hint" style={{ padding: '0 0 4px' }}>Share of total radiance · {fmtSci(contribTotal)} mW mm⁻² sr⁻¹ summed over {px.emission_nm[exKey].length} bands</div>
            {contrib.map((c) => (
              <div key={c.name} className="crow" title={`${c.name}: ${fmtSci(c.sum)} mW mm⁻² sr⁻¹`}>
                <i style={{ background: c.color }} /><span className="cname">{c.name}</span>
                <span className="cbar"><b style={{ width: `${Math.max(1, c.pct)}%`, background: c.color }} /></span>
                <span className="cpct mono">{c.pct < 0.1 ? '<0.1' : c.pct.toFixed(c.pct < 10 ? 1 : 0)} %</span>
                <span className="cabs mono">{fmtSci(c.sum)}</span>
              </div>))}
          </div>}
          <div className="plot-title">Spectra per excitation <small>ADU</small></div>
          <div style={{ padding: '0 8px 4px' }}><LineChart series={specSeries} height={130} strip xLabel="emission nm" yLabel="ADU" legend /></div>
          {eemd && eemd.exs.length > 1 && <>
            <div className="plot-title">Excitation–emission matrix <small>pixel radiance</small></div>
            <div style={{ padding: '0 8px 4px' }}><Heatmap data={eemd.data} xs={eemd.grid} ys={eemd.exs} height={150} xLabel="emission nm" yLabel="ex nm" marks={[{ y: ex, color: wavelengthToHex(ex) }]} /></div></>}
          <div className="plot-title">Ground truth <small>{Object.keys(px.ground_truth).length} fields</small>
            {Object.keys(px.ground_truth).length > 8 && <button className="btn ghost sm" style={{ marginLeft: 'auto' }} onClick={() => setAllGt(!allGt)}>{allGt ? 'less' : 'all'}</button>}</div>
          <dl className="kv">{Object.entries(px.ground_truth).slice(0, allGt ? undefined : 8).map(([k, v]) => <Fragment key={k}><dt>{k.replace(/_/g, ' ')}</dt><dd>{v == null ? '—' : typeof v === 'number' ? fmtSig(v, gtUnit(k)) : String(v)}</dd></Fragment>)}</dl>
        </>}
      </>}
      <div className="plot-title">Capture</div>
      <dl className="kv">
        <dt>size</dt><dd>{meta.width}×{meta.height}</dd>
        <dt>excitations</dt><dd>{meta.excitations_nm.length}</dd>
        <dt>bands @ {ex} nm</dt><dd>{meta.emission_nm[exKey]?.length}</dd>
        <dt>exposure / power</dt><dd>{meta.exposure_ms[exKey]} ms · {meta.power_mW[exKey]} mW</dd>
        <dt>saturated</dt><dd style={{ color: meta.saturated_fraction[exKey] > 0.001 ? 'var(--warn)' : undefined }}>{(meta.saturated_fraction[exKey] * 100).toFixed(3)} %</dd>
        {Object.entries(meta.timings_s).map(([k, v]) => <Fragment key={k}><dt>t · {k}</dt><dd>{v.toFixed(2)} s</dd></Fragment>)}
      </dl>
      {meta.warnings.length > 0 && <div style={{ padding: '0 12px 16px', display: 'grid', gap: 4 }}>{meta.warnings.map((w, i) => <div key={i} className="hint" style={{ color: 'var(--warn)' }}>⚠ {w}</div>)}</div>}
    </div>
  )
}

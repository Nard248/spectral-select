import { useMemo } from 'react'
import { FileJson, FileSpreadsheet } from 'lucide-react'
import type { CaptureMeta, EemResult, RoiSpec } from '../api/types'
import { useAnalysis } from './store'
import { isRegion, type NamedRoi, type RegionSpec } from './types'
import { useEemMulti, useRoiStats, type EemItem } from './hooks'
import { alignEem, atEmission, brightestBand, componentLabel, diffGrid, log2RatioGrid } from './eemMath'
import { XYChart, EemHeatmap, fmtVal, type XYSeries } from '../ui/plots'
import { StackedArea } from '../ui/charts'
import { Seg, Sel, Toggle, Tip } from '../ui/controls'
import { wavelengthToHex } from '../lib/color'
import { describe } from './roi'
import { downloadText, eemToCsv, roiStatsToCsv, safeName, toJson, type ExportContext } from './exportData'
import { fluorophoreMap } from '../state/presets'
import { useStudio } from '../state/store'
import { ChartBox, Busy, WsEmpty } from './parts'

const PALETTE = ['#4ef28a', '#ff9b4a', '#b48cff', '#35d8ff', '#ff6bb0', '#e8e35a', '#6b9dff', '#ff6b5e']

export function EemPanel({ meta, all, ex }: { meta: CaptureMeta; all: CaptureMeta[]; ex: number | null }) {
  const rois = useAnalysis((s) => s.rois)
  const selectedId = useAnalysis((s) => s.selectedId)
  const select = useAnalysis((s) => s.select)
  const norm = useAnalysis((s) => s.norm)
  const stat = useAnalysis((s) => s.stat)
  const eem = useAnalysis((s) => s.eem)
  const setEem = useAnalysis((s) => s.setEem)
  const expState = useStudio((s) => s.exp)
  const fl = useMemo(() => fluorophoreMap(expState), [expState])

  const regions = useMemo(() => rois.filter((r): r is NamedRoi & { spec: RegionSpec } => isRegion(r.spec)), [rois])
  const primary = regions.find((r) => r.id === selectedId) ?? regions[0] ?? null
  const comparing = eem.mode === 'diff' || eem.mode === 'ratio'
  const roiB = comparing && !eem.captureB ? (regions.find((r) => r.id === eem.roiB && r.id !== primary?.id) ?? regions.find((r) => r.id !== primary?.id) ?? null) : null
  const capB = comparing && eem.captureB ? all.find((c) => c.id === eem.captureB) ?? null : null

  const items = useMemo<EemItem[]>(() => {
    if (!primary) return []
    const out: EemItem[] = [{ key: 'A', captureId: meta.id, roi: primary.spec as RoiSpec }]
    if (eem.mode === 'overlay') regions.filter((r) => r.visible && r.id !== primary.id).forEach((r) => out.push({ key: `roi:${r.id}`, captureId: meta.id, roi: r.spec as RoiSpec }))
    if (comparing && roiB) out.push({ key: 'B', captureId: meta.id, roi: roiB.spec as RoiSpec })
    if (comparing && capB) out.push({ key: 'B', captureId: capB.id, roi: primary.spec as RoiSpec })
    return out
  }, [primary, regions, eem.mode, comparing, roiB, capB, meta.id])

  const res = useEemMulti(items, norm, stat)
  const A: EemResult | undefined = res.data.A
  const B: EemResult | undefined = res.data.B
  const stats = useRoiStats(meta.id, primary ? (primary.spec as RoiSpec) : null, norm, stat, eem.showComponents)

  const exs = A?.excitations_nm ?? []
  const sliceEx = eem.sliceEx != null && exs.some((e) => Math.abs(e - eem.sliceEx!) < 0.5) ? eem.sliceEx : (ex != null && exs.includes(ex) ? ex : exs[0] ?? null)
  const sliceEm = eem.sliceEm ?? (A && sliceEx != null ? brightestBand(A, sliceEx) : null)

  const bLabel = capB ? `${primary?.name} @ ${capB.name}` : roiB?.name ?? 'B'
  const bColor = roiB?.color ?? '#e8ecf2'

  const align = useMemo(() => (A && B ? alignEem(A, B) : null), [A, B])
  const heat = useMemo(() => {
    if (!A) return null
    if (comparing && align) {
      const z = eem.mode === 'diff' ? diffGrid(align) : log2RatioGrid(align)
      return { z, xs: align.grid, ys: align.exs, cmap: 'diverging' as const, unit: eem.mode === 'diff' ? `Δ ${A.unit}` : 'log₂ A/B' }
    }
    return { z: A.grid_value, xs: A.grid_emission_nm, ys: A.excitations_nm, cmap: eem.cmap, unit: A.unit }
  }, [A, align, comparing, eem.mode, eem.cmap])

  const unit = A?.unit ?? ''
  const emSeries = useMemo<XYSeries[]>(() => {
    if (!A || sliceEx == null) return []
    const key = String(A.excitations_nm.find((e) => Math.abs(e - sliceEx) < 0.5) ?? sliceEx)
    const out: XYSeries[] = []
    if (eem.mode === 'single') {
      A.excitations_nm.forEach((e) => {
        const k = String(e), sel = k === key
        out.push({ name: `${e} nm`, x: A.emission_nm[k], y: A.value[k], std: sel && eem.showStd ? A.std[k] : null, color: wavelengthToHex(e), width: sel ? 2.3 : 1, opacity: sel ? 1 : 0.5 })
      })
      return out.sort((a, b) => (a.width ?? 1) - (b.width ?? 1))
    }
    if (eem.mode === 'overlay') {
      const mk = (name: string, color: string, r: EemResult) => ({ name, x: r.emission_nm[key], y: r.value[key], std: eem.showStd ? r.std[key] : null, color, width: 2 })
      if (primary) out.push(mk(primary.name, primary.color, A))
      regions.filter((r) => r.id !== primary?.id).forEach((r) => { const d = res.data[`roi:${r.id}`]; if (d && d.emission_nm[key]) out.push(mk(r.name, r.color, d)) })
      return out
    }
    if (primary) out.push({ name: primary.name, x: A.emission_nm[key], y: A.value[key], std: eem.showStd ? A.std[key] : null, color: primary.color, width: 2 })
    if (B && B.emission_nm[key]) out.push({ name: bLabel, x: B.emission_nm[key], y: B.value[key], std: eem.showStd ? B.std[key] : null, color: bColor, width: 2 })
    if (B && B.emission_nm[key]) {
      const xs = A.emission_nm[key]
      const yb = xs.map((x) => atEmission(B, sliceEx, x).v)
      out.push({ name: eem.mode === 'diff' ? 'A − B' : 'A / B', x: xs, y: xs.map((_, i) => (eem.mode === 'diff' ? A.value[key][i] - yb[i] : yb[i] ? A.value[key][i] / yb[i] : NaN)), color: '#ffffff', dash: '4 3', width: 1.5, axis: 'r' })
    }
    return out
  }, [A, B, res.data, eem.mode, eem.showStd, sliceEx, primary, regions, bLabel, bColor])

  const exSeries = useMemo<XYSeries[]>(() => {
    if (!A || sliceEm == null) return []
    const mk = (name: string, color: string, r: EemResult): XYSeries => {
      const v = r.excitations_nm.map((e) => atEmission(r, e, sliceEm))
      return { name, x: r.excitations_nm, y: v.map((q) => q.v), std: eem.showStd ? v.map((q) => q.s) : null, color, width: 2, points: true }
    }
    if (eem.mode === 'single') return [mk(primary?.name ?? 'ROI', primary?.color ?? '#35d8ff', A)]
    if (eem.mode === 'overlay') return [mk(primary!.name, primary!.color, A), ...regions.filter((r) => r.id !== primary?.id).flatMap((r) => (res.data[`roi:${r.id}`] ? [mk(r.name, r.color, res.data[`roi:${r.id}`])] : []))]
    const out = [mk(primary?.name ?? 'A', primary?.color ?? '#35d8ff', A)]
    if (B) out.push(mk(bLabel, bColor, B))
    return out
  }, [A, B, res.data, eem.mode, eem.showStd, sliceEm, primary, regions, bLabel, bColor])

  const ctx = (): ExportContext => ({ captureId: meta.id, experimentName: meta.name, norm: A?.norm ?? norm, roiName: primary?.name, roi: primary?.spec })
  const stem = safeName(`${meta.name}_${primary?.name ?? 'roi'}_eem`)
  const gtEntries = stats.data ? Object.entries(stats.data.ground_truth).filter(([k]) => !['material_index', 'object_index'].includes(k)) : []
  const compLayers = useMemo(() => {
    const comp = stats.data?.components?.[String(sliceEx)]
    const exKey = String(sliceEx)
    if (!comp || !stats.data) return null
    let pi = 0
    const layers = Object.entries(comp).map(([name, y]) => {
      const dyeKey = name.split('@')[0].replace(/^(dye|secondary):/, '')
      const color = name.startsWith('dye:') && fl[dyeKey]?.display_color ? fl[dyeKey].display_color! : PALETTE[pi++ % PALETTE.length]
      return { name: componentLabel(name), color, y: Array.from(y as ArrayLike<number>) }
    }).filter((l) => l.y.some((v) => v > 1e-9))
    return { layers, x: stats.data.emission_nm[exKey] }
  }, [stats.data, sliceEx, fl])

  const rowsTable = A && sliceEx != null ? (() => {
    const k = String(A.excitations_nm.find((e) => Math.abs(e - sliceEx) < 0.5) ?? sliceEx)
    return (A.emission_nm[k] ?? []).map((nm, i) => ({ nm, v: A.value[k][i], s: A.std[k][i] }))
  })() : []

  if (!regions.length) return (
    <WsEmpty icon="eem" title="Draw a region to get its EEM">
      Use the <b style={{ display: 'inline', color: 'var(--accent)' }}>disk</b>, <b style={{ display: 'inline', color: 'var(--accent)' }}>rectangle</b>, ellipse or polygon tool above the image (shortcuts <span className="kbd">D</span> <span className="kbd">R</span> <span className="kbd">E</span> <span className="kbd">G</span>).
      The excitation–emission matrix of the region — averaged over its pixels — appears here, with slices, ±std and A/B comparison.
    </WsEmpty>)

  return (
    <div className="scroll">
      <div className="ctrls" style={{ paddingBottom: 2 }}>
        <Seg small value={eem.mode} onChange={(mode) => setEem({ mode })} options={[
          { value: 'single', label: 'Single', title: 'One ROI: full EEM and per-excitation spectra' },
          { value: 'overlay', label: 'Overlay', title: 'Every visible ROI on the same chart' },
          { value: 'diff', label: 'A − B', title: 'Difference between ROI A and B (or the same ROI in two captures)' },
          { value: 'ratio', label: 'A / B', title: 'Ratio A / B on a log2 scale' },
        ]} />
        <Toggle value={eem.log} label="log" onChange={(log) => setEem({ log })} tip="Compressive (log-like) colour scale for the heatmap." />
        <Toggle value={eem.contours} label="contours" onChange={(contours) => setEem({ contours })} />
        <Toggle value={eem.showStd} label="±std" onChange={(showStd) => setEem({ showStd })} tip="Shade ±1 standard deviation across the ROI pixels." />
      </div>
      <div className="chips2">
        {regions.map((r) => (
          <button key={r.id} className={`chip2${r.id === primary?.id ? ' on' : ''}`} style={{ ['--c' as string]: r.color, opacity: r.visible ? 1 : 0.5 }} onClick={() => select(r.id)} title={describe(r.spec)}><i />{r.name}</button>))}
      </div>
      {comparing && (
        <div className="ctrls" style={{ paddingTop: 0 }}>
          <span className="grp">B =</span>
          <div style={{ minWidth: 190, flex: 1 }}>
            <Sel value={eem.captureB ? `cap:${eem.captureB}` : `roi:${roiB?.id ?? ''}`}
              options={[
                ...regions.filter((r) => r.id !== primary?.id).map((r) => ({ value: `roi:${r.id}`, label: `ROI: ${r.name}` })),
                ...all.filter((c) => c.id !== meta.id).map((c) => ({ value: `cap:${c.id}`, label: `${primary?.name ?? 'ROI'} in: ${c.name} (${c.mode}) ${new Date(c.created).toLocaleTimeString()}` })),
              ]}
              onChange={(v) => (v.startsWith('cap:') ? setEem({ captureB: v.slice(4), roiB: null }) : setEem({ captureB: null, roiB: v.slice(4) }))} />
          </div>
          {!roiB && !capB && <span className="hint" style={{ color: 'var(--warn)' }}>Draw a second ROI, or pick another capture, to compare.</span>}
        </div>)}

      <div className="asec">
        <div className="h">EEM <small>{A ? `${A.n_pixels} px · ${stat} · ${eem.mode === 'diff' ? 'A − B' : eem.mode === 'ratio' ? 'log₂ A/B' : unit}` : ''}</small><span className="sp" /><Busy on={res.loading} /></div>
        <div className="pad">
          {res.error && <div className="note err">{res.error}</div>}
          {heat && <ChartBox><EemHeatmap z={heat.z} xs={heat.xs} ys={heat.ys} cmap={heat.cmap} log={eem.log && !comparing} contours={eem.contours} unit={heat.unit} height={Math.min(310, 140 + heat.ys.length * 16)}
            marks={{ ex: sliceEx, em: sliceEm }} onPick={(e, m) => setEem({ sliceEx: e, sliceEm: m })} /></ChartBox>}
          {!A && !res.error && <div className="hint" style={{ padding: 20, textAlign: 'center' }}>computing…</div>}
          {comparing && !B && A && <div className="note warn">Pick B to see the {eem.mode === 'diff' ? 'difference' : 'ratio'} map.</div>}
        </div>
      </div>

      <div className="asec">
        <div className="h">Emission spectra <small>at {sliceEx ?? '—'} nm excitation{eem.mode === 'single' ? ' (others faint)' : ''}</small></div>
        <div className="pad"><ChartBox>
          <XYChart series={emSeries} height={170} xLabel="emission (nm)" yLabel={unit} yLabelR={comparing ? (eem.mode === 'diff' ? `A − B (${unit})` : 'A / B') : undefined} strip legend={eem.mode !== 'single'} />
        </ChartBox></div>
        <div className="chips2" style={{ paddingTop: 2 }}>
          {exs.map((e) => <button key={e} className={`chip2 mono${sliceEx != null && Math.abs(e - sliceEx) < 0.5 ? ' on' : ''}`} style={{ ['--c' as string]: wavelengthToHex(e) }} onClick={() => setEem({ sliceEx: e })}><i />{e}</button>)}
        </div>
      </div>

      <div className="asec">
        <div className="h">Excitation spectrum <small>at {sliceEm != null ? `${sliceEm.toFixed(0)} nm emission` : '—'}</small>
          <span className="sp" />
          {A && sliceEx != null && <Tip text="Click the heatmap to pick the excitation row and emission column of these slices."><span className="hint">click the EEM to pick</span></Tip>}</div>
        <div className="pad"><ChartBox>
          <XYChart series={exSeries} height={150} xLabel="excitation (nm)" yLabel={unit} legend={eem.mode !== 'single'} zeroBase />
        </ChartBox></div>
      </div>

      {A && <div className="asec">
        <div className="h">Per-band statistics <small>{sliceEx} nm · {stat} ± std over {A.n_pixels} px</small></div>
        <div className="tscroll"><table className="stat"><thead><tr><th>λ nm</th><th>{stat}</th><th>± std</th><th>CV</th></tr></thead>
          <tbody>{rowsTable.map((r) => <tr key={r.nm}><td style={{ color: wavelengthToHex(r.nm) }}>{r.nm.toFixed(0)}</td><td>{fmtVal(r.v)}</td><td>{fmtVal(r.s)}</td><td>{r.v ? `${((100 * r.s) / Math.abs(r.v)).toFixed(1)} %` : '—'}</td></tr>)}</tbody></table></div>
      </div>}

      <div className="asec">
        <div className="h">Ground truth in ROI <small>{primary?.name}</small><span className="sp" />
          <Toggle value={eem.showComponents} label="components" onChange={(showComponents) => setEem({ showComponents })} tip="Also average the noise-free component decomposition over ≤ 64 pixels of the ROI." /></div>
        {gtEntries.length > 0 ? (
          <div className="tscroll"><table className="stat"><thead><tr><th>layer</th><th>mean</th><th>± std</th><th>min</th><th>max</th></tr></thead>
            <tbody>{gtEntries.map(([k, v]) => <tr key={k}><td title={k}>{k.replace(/_/g, ' ')}</td><td>{fmtVal(v.mean)}</td><td>{fmtVal(v.std)}</td><td>{fmtVal(v.min)}</td><td>{fmtVal(v.max)}</td></tr>)}</tbody></table></div>)
          : <div className="hint" style={{ padding: '2px 12px' }}>{stats.loading ? 'loading…' : stats.error ?? 'No numeric ground truth inside this ROI.'}</div>}
        {eem.showComponents && compLayers && compLayers.layers.length > 0 && (
          <div style={{ padding: '10px 10px 0' }}>
            <div className="hint" style={{ padding: '0 2px 4px' }}>Averaged components at {sliceEx} nm — layer of origin after “·”.</div>
            <StackedArea x={compLayers.x} layers={compLayers.layers} height={170} xLabel="emission nm" yLabel="radiance" />
          </div>)}
      </div>

      <div className="exports">
        <span className="hint">Export this ROI</span>
        <button className="btn sm" disabled={!A} onClick={() => A && downloadText(`${stem}.csv`, eemToCsv(A, ctx()))}><FileSpreadsheet size={13} />CSV</button>
        <button className="btn sm" disabled={!A} onClick={() => A && downloadText(`${stem}.json`, toJson('eem', ctx(), { eem: A, ground_truth: stats.data?.ground_truth ?? null, b: B ?? null }), 'application/json')}><FileJson size={13} />JSON</button>
        <button className="btn sm ghost" disabled={!stats.data} onClick={() => stats.data && downloadText(`${stem}_groundtruth.csv`, roiStatsToCsv(stats.data, ctx()))}>GT stats CSV</button>
      </div>
      <div style={{ height: 14 }} />
    </div>
  )
}

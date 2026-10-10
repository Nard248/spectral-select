import { useCallback, useEffect, useMemo, useState } from 'react'
import { Play, Pause, ChevronLeft, ChevronRight, Download, MousePointer2, BoxSelect, Columns2, Image as ImageIcon } from 'lucide-react'
import { useStudio } from '../state/store'
import { getEngine, type DownloadFormat } from '../api/client'
import { serialize } from '../state/io'
import { ImageView, type ImageStyle, type Roi } from './ImageView'
import { PixelPanel } from './PixelPanel'
import { useCube, useGt } from './data'
import { Empty, MenuButton, Num, Seg, Sel, Toggle } from '../ui/controls'
import { CMAP_NAMES, cssGradient, type CmapName } from '../ui/colormaps'
import { wavelengthToHex } from '../lib/color'
import { toast } from '../ui/toast'
import type { CaptureMeta } from '../api/types'

function exportCapture(meta: CaptureMeta, fmt: DownloadFormat) {
  const eng = getEngine()
  if (fmt === 'experiment_json' && eng?.kind === 'mock') {
    const blob = new Blob([serialize(useStudio.getState().exp)], { type: 'application/json' })
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `${meta.name}.sfx.json`; a.click(); return
  }
  if (!eng || eng.kind === 'mock') { toast('Exports are produced by the Python engine — not available in mock mode', 'error'); return }
  const a = document.createElement('a'); a.href = eng.downloadUrl(meta.id, fmt); a.download = ''; document.body.appendChild(a); a.click(); a.remove()
}

export function Explorer() {
  const captures = useStudio((s) => s.captures)
  const previewMeta = useStudio((s) => s.previewMeta)
  const activeId = useStudio((s) => s.activeCaptureId)
  const compareId = useStudio((s) => s.compareCaptureId)
  const setMode = useStudio((s) => s.setMode)
  const all = useMemo(() => (previewMeta && !captures.some((c) => c.id === previewMeta.id) ? [...captures, previewMeta] : captures), [captures, previewMeta])
  const meta = all.find((c) => c.id === activeId) ?? all[0] ?? null
  const metaB = compareId ? all.find((c) => c.id === compareId) ?? null : null

  const [exIdx, setExIdx] = useState(0)
  const [band, setBand] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [style, setStyle] = useState<ImageStyle>({ mode: 'band', cmap: 'viridis', pLo: 0.01, pHi: 0.995, log: false, globalScale: false })
  const [tool, setTool] = useState<'pixel' | 'roi'>('pixel')
  const [pixel, setPixel] = useState<{ x: number; y: number } | null>(null)
  const [roi, setRoi] = useState<Roi | null>(null)
  const [gtName, setGtName] = useState<string | null>(null)
  const [hover, setHover] = useState<{ x: number; y: number; v: number } | null>(null)
  const [range, setRange] = useState<[number, number]>([0, 1])
  const onRange = useCallback((lo: number, hi: number) => setRange([lo, hi]), [])

  const exs = meta?.excitations_nm ?? []
  const ex = exs[Math.min(exIdx, exs.length - 1)] ?? null
  const emission = meta && ex != null ? meta.emission_nm[String(ex)] ?? [] : []
  const { data: cube, loading, error } = useCube(meta?.id ?? null, ex)
  const exB = metaB ? (metaB.excitations_nm.includes(ex ?? -1) ? ex : metaB.excitations_nm[0]) : null
  const { data: cubeB } = useCube(metaB?.id ?? null, exB)
  const { data: gt } = useGt(meta?.id ?? null, gtName)

  useEffect(() => { setExIdx(0); setBand(0); setPixel(null); setRoi(null); setGtName(null) }, [meta?.id])
  const pickRequest = useStudio((s) => s.pickRequest)
  useEffect(() => {
    if (!pickRequest || !meta) return
    const i = pickRequest.exNm != null ? meta.excitations_nm.findIndex((e) => Math.abs(e - pickRequest.exNm!) < 0.5) : -1
    if (i >= 0) setExIdx(i)
    setPixel({ x: pickRequest.x, y: pickRequest.y })
    if (pickRequest.band != null) setBand(pickRequest.band)
    useStudio.setState({ pickRequest: null })
  }, [pickRequest, meta])
  useEffect(() => { if (band >= emission.length && emission.length) setBand(emission.length - 1) }, [emission.length, band])
  useEffect(() => {
    if (!playing || !emission.length) return
    const t = setInterval(() => setBand((b) => (b + 1) % emission.length), 130)
    return () => clearInterval(t)
  }, [playing, emission.length])
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)) return
      if (e.key === 'ArrowRight') { setBand((b) => Math.min(emission.length - 1, b + 1)); e.preventDefault() }
      else if (e.key === 'ArrowLeft') { setBand((b) => Math.max(0, b - 1)); e.preventDefault() }
      else if (e.key === 'ArrowUp') { setExIdx((i) => Math.max(0, i - 1)); e.preventDefault() }
      else if (e.key === 'ArrowDown') { setExIdx((i) => Math.min(exs.length - 1, i + 1)); e.preventDefault() }
      else if (e.key === ' ') { setPlaying((p) => !p); e.preventDefault() }
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [emission.length, exs.length])

  if (!meta) return (
    <div className="explore" style={{ gridTemplateColumns: '1fr' }}>
      <div className="col" style={{ justifyContent: 'center' }}>
        <Empty title="No captures yet"><div style={{ maxWidth: 360, margin: '0 auto' }}>Run <b style={{ display: 'inline', color: 'var(--accent)' }}>Preview</b> or <b style={{ display: 'inline', color: 'var(--accent)' }}>Capture</b> from the sequence bar and the resulting 4D cube appears here.</div>
          <button className="btn primary" style={{ marginTop: 14 }} onClick={() => setMode('design')}>Back to Design</button></Empty>
      </div>
    </div>)

  const bandNm = emission[band]
  const pick = (id: string) => useStudio.setState({ activeCaptureId: id })

  return (
    <div className="explore" style={{ gridTemplateColumns: undefined }}>
      {/* captures */}
      <div className="col">
        <div className="ptitle">Captures</div>
        <div className="scroll">
          {all.slice().reverse().map((c) => (
            <div key={c.id} className={`capitem${c.id === meta.id ? ' sel' : ''}`} onClick={() => pick(c.id)}>
              <div className="t">{c.name}<span className="tag">{c.mode}</span></div>
              <small>{c.width}×{c.height} · {c.excitations_nm.length} ex · {new Date(c.created).toLocaleTimeString()}</small>
              <div style={{ display: 'flex', gap: 3, marginTop: 3 }}>{c.excitations_nm.map((e) => <i key={e} style={{ width: 9, height: 9, borderRadius: 3, background: wavelengthToHex(e), display: 'inline-block' }} />)}</div>
            </div>))}
        </div>
        <div style={{ padding: 10, borderTop: '1px solid var(--line)', display: 'grid', gap: 8 }}>
          <div className="row" style={{ gridTemplateColumns: '70px 1fr' }}><label>Compare</label>
            <Sel value={compareId ?? ''} options={[{ value: '', label: '— off —' }, ...all.filter((c) => c.id !== meta.id).map((c) => ({ value: c.id, label: `${c.name} (${c.mode})` }))]} onChange={(v) => useStudio.setState({ compareCaptureId: v || null })} /></div>
          <div className="row" style={{ gridTemplateColumns: '70px 1fr' }}><label>GT overlay</label>
            <Sel value={gtName ?? ''} options={[{ value: '', label: '— none —' }, ...meta.ground_truth_layers.map((n) => ({ value: n, label: n }))]} onChange={(v) => setGtName(v || null)} /></div>
          <MenuButton label={<><Download size={14} />Export</>} right>
            {(close) => <>{(['spectradata_pkl', 'npz', 'tiff_zip', 'ground_truth_zip', 'experiment_json'] as DownloadFormat[]).map((f) => <button key={f} className="mi" onClick={() => { close(); exportCapture(meta, f) }}>{f.replace(/_/g, ' ')}</button>)}</>}
          </MenuButton>
        </div>
      </div>

      {/* image */}
      <div className="col mid">
        <div className="exbar">
          <span className="ptitle" style={{ padding: 0 }}>Excitation</span>
          <div className="tabs">{exs.map((e, i) => <button key={e} className={`tab${i === exIdx ? ' on' : ''}`} style={{ ['--c' as string]: wavelengthToHex(e) }} onClick={() => setExIdx(i)}><i />{e}</button>)}</div>
          <Seg small value={tool} onChange={setTool} options={[{ value: 'pixel', label: <MousePointer2 size={13} />, title: 'Pick pixel' }, { value: 'roi', label: <BoxSelect size={13} />, title: 'Draw ROI' }]} />
          <Seg small value={style.mode} onChange={(mode) => setStyle({ ...style, mode })} options={[{ value: 'band', label: 'Band' }, { value: 'true', label: 'True colour' }, { value: 'false', label: 'False colour' }]} />
        </div>
        <div style={{ position: 'relative', flex: 1, minHeight: 0, display: 'flex' }}>
          {loading && !cube && <div className="splash" style={{ background: 'transparent' }}><div><div className="spinner" />loading cube…</div></div>}
          {error && <div className="splash" style={{ color: 'var(--err)' }}>{error}</div>}
          {cube && <div style={{ display: 'flex', flex: 1, minWidth: 0 }}>
            <div style={{ flex: 1, minWidth: 0, display: 'flex', position: 'relative' }}>
              <ImageView cube={cube} emission={emission} band={band} style={style} gt={gt} pixel={pixel} roi={roi} tool={tool}
                onPixel={(x, y) => setPixel({ x, y })} onRoi={setRoi} onHover={(x, y, v) => setHover({ x, y, v })} onRange={onRange}
                label={`${meta.name} · ${meta.mode}`} />
            </div>
            {metaB && cubeB && exB != null && <div style={{ flex: 1, minWidth: 0, display: 'flex', borderLeft: '1px solid var(--line)' }}>
              <ImageView cube={cubeB} emission={metaB.emission_nm[String(exB)]} band={Math.min(band, cubeB.bands - 1)} style={style} pixel={pixel} roi={roi} tool={tool}
                onPixel={(x, y) => setPixel({ x, y })} onRoi={setRoi} label={`${metaB.name} · ${metaB.mode}`} />
            </div>}
          </div>}
          <div className="glass" style={{ position: 'absolute', left: 12, bottom: 12, padding: '6px 10px', display: 'flex', gap: 12, alignItems: 'center', fontSize: 11.5 }}>
            {style.mode === 'band' ? <>
              <span className="cmap-bar" style={{ background: cssGradient(style.cmap) }} />
              <span className="mono">{range[0].toFixed(0)} – {range[1].toFixed(0)} ADU</span></> : <span className="hint">{style.mode === 'true' ? 'Emission colours summed per pixel' : 'Three bands mapped to RGB'}</span>}
            {hover && <span className="mono" style={{ color: 'var(--dim)' }}>({hover.x}, {hover.y}) = {hover.v.toFixed(0)}</span>}
          </div>
        </div>
        <div className="scrub">
          <button className="btn icon" onClick={() => setPlaying(!playing)} title="Play bands [Space]">{playing ? <Pause size={15} /> : <Play size={15} />}</button>
          <button className="btn ghost icon sm" onClick={() => setBand(Math.max(0, band - 1))}><ChevronLeft size={15} /></button>
          <input type="range" min={0} max={Math.max(0, emission.length - 1)} value={band} onChange={(e) => setBand(+e.target.value)} />
          <button className="btn ghost icon sm" onClick={() => setBand(Math.min(emission.length - 1, band + 1))}><ChevronRight size={15} /></button>
          <span className="mono" style={{ minWidth: 112, textAlign: 'right', color: bandNm != null ? wavelengthToHex(bandNm) : undefined }}>band {band + 1}/{emission.length} · {bandNm?.toFixed(0)} nm</span>
        </div>
        <div className="exbar" style={{ borderTop: '1px solid var(--line)', borderBottom: 0 }}>
          <div style={{ width: 96 }}><Sel value={style.cmap} options={CMAP_NAMES as CmapName[]} onChange={(cmap) => setStyle({ ...style, cmap })} /></div>
          <span className="hint">contrast</span>
          <div style={{ width: 70 }}><Num value={style.pLo * 100} min={0} max={50} step={0.5} unit="%" precision={1} bar={false} onChange={(v) => setStyle({ ...style, pLo: v / 100 })} /></div>
          <div style={{ width: 70 }}><Num value={style.pHi * 100} min={50} max={100} step={0.1} unit="%" precision={1} bar={false} onChange={(v) => setStyle({ ...style, pHi: v / 100 })} /></div>
          <Toggle value={style.log} label="log" onChange={(log) => setStyle({ ...style, log })} />
          <Toggle value={style.globalScale} label="global scale" onChange={(globalScale) => setStyle({ ...style, globalScale })} />
          <span style={{ marginLeft: 'auto', display: 'inline-flex', gap: 6, alignItems: 'center', color: 'var(--faint)' }}><Columns2 size={13} /><ImageIcon size={13} /><span className="hint">←→ bands · ↑↓ excitation</span></span>
        </div>
      </div>

      {/* analysis */}
      <div className="col">
        <div className="ptitle">Analysis</div>
        {ex != null && <PixelPanel meta={meta} pixel={pixel} ex={ex} roi={roi} cube={cube} />}
      </div>
    </div>
  )
}

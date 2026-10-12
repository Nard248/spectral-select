import { useEffect, useMemo, useRef, useState } from 'react'
import type { Experiment } from '../generated/experiment'
import type { VolumeInfo } from '../api/types'
import type { CaptureMetaX } from './types'
import { useCaptureExperiment, useVolume, useVolumeInfos } from './hooks'
import { useStudio } from '../state/store'
import { Num, Sel, Seg, Toggle } from '../ui/controls'
import { useWidth } from '../ui/charts'
import { fmtTick, fmtVal, niceTicks } from '../ui/plots'
import { cssGradient, type CmapName } from '../ui/colormaps'
import { wavelengthToHex } from '../lib/color'
import { autoScale, depthProfiles, geometry, locate, renderXY, renderXZ, shapeMatches, type VolGeom } from './volumeGeom'
import { stackAt } from './blockGeometry'
import { downloadText, safeName, toJson } from './exportData'
import { Busy, WsEmpty } from './parts'
import { FileJson } from 'lucide-react'

const LAYER_PALETTE = ['#7d8da6', '#4ef28a', '#ffb547', '#ff6bb0', '#b48cff', '#35d8ff', '#e8e35a', '#ff7a5e']
const KIND_LABEL = { fluence: 'Fluence', absorbed: 'Absorbed power', source: 'Emission source' } as const
const KIND_TIP = { fluence: 'Excitation fluence rate inside the block (track-length estimator).', absorbed: 'Power absorbed per voxel by all absorbers.', source: 'Where a chosen fluorophore emits: absorbed excitation × yield, per layer and dye.' } as const

function useDpr() { return typeof window !== 'undefined' ? Math.min(2, window.devicePixelRatio || 1) : 1 }

export function LightField({ meta }: { meta: CaptureMetaX }) {
  const infosQ = useVolumeInfos(meta.id, meta.volumes)
  const infos = useMemo(() => infosQ.data ?? [], [infosQ.data])
  const storeExp = useStudio((s) => s.exp)
  const capExpQ = useCaptureExperiment(meta.id)

  const objects = useMemo(() => [...new Set(infos.map((v) => v.object_id))], [infos])
  const [objectId, setObject] = useState<string | null>(null)
  const [kind, setKind] = useState<VolumeInfo['kind']>('fluence')
  const [exNm, setEx] = useState<number | null>(null)
  const [channel, setChannel] = useState<string | null>(null)
  const [log, setLog] = useState(true)
  const [decades, setDecades] = useState(3)
  const [cmap, setCmap] = useState<CmapName>('inferno')
  const [zFrac, setZFrac] = useState(0.5)
  const [yMm, setY] = useState<number | null>(null)
  useEffect(() => { setY(null) }, [channel, kind, objectId])
  const [exagUser, setExag] = useState<number | null>(null)

  const obj = objectId && objects.includes(objectId) ? objectId : objects[0] ?? null
  const kinds = infos.filter((v) => v.object_id === obj).map((v) => v.kind)
  const k: VolumeInfo['kind'] = kinds.includes(kind) ? kind : kinds[0] ?? 'fluence'
  const info = infos.find((v) => v.object_id === obj && v.kind === k) ?? null
  const exs = info?.excitations_nm ?? []
  const ex = exNm != null && exs.some((e) => Math.abs(e - exNm) < 0.5) ? exNm : exs[0] ?? null
  const ch = k === 'source' ? (channel && info?.channels.includes(channel) ? channel : info?.channels[0] ?? null) : null

  // geometry source: the capture's own experiment when the engine can supply it, else the open one (if it matches)
  const shape = useMemo(() => {
    for (const e of [capExpQ.data, storeExp] as (Experiment | null | undefined)[]) {
      const o = e?.objects.find((q) => q.id === obj)
      if (o && o.shape.kind === 'layered_block' && info && shapeMatches(info, o.shape)) return { shape: o.shape, exp: e! }
    }
    return null
  }, [capExpQ.data, storeExp, obj, info])
  const g = useMemo(() => (info && shape ? geometry(info, shape.shape) : null), [info, shape])
  const vol = useVolume(meta.id, info && ex != null ? { object: info.object_id, kind: k, ex, channel: ch ?? undefined } : null)
  const data = vol.data && g && vol.data.dims[0] === g.nx && vol.data.dims[1] === g.nz && vol.data.dims[2] === g.nd ? vol.data.data : null

  const sc = useMemo(() => (data ? autoScale(data, log, decades, cmap) : null), [data, log, decades, cmap])
  const layerColors = useMemo(() => info?.layer_names.map((n, i) => {
    const l = shape?.shape.layers[i]; const m = l && shape?.exp.materials.find((q) => q.id === l.material_id)
    void n
    return m?.display_color ?? LAYER_PALETTE[i % LAYER_PALETTE.length]
  }) ?? [], [info, shape])
  const profiles = useMemo(() => (g && data ? depthProfiles(g, data) : []), [g, data])

  if (!infos.length) return (
    <div className="lf"><WsEmpty icon="light" title={infosQ.loading ? 'Loading volumes…' : 'No 3D light field in this capture'}>
      Light-field volumes exist for captures that contain a <b style={{ display: 'inline', color: 'var(--accent)' }}>layered block</b> (substrate, fluorescent layer, cover…). Load the “Layered wedge phantom” or “Tissue under collagen” preset and capture again.</WsEmpty></div>)

  // default plane height: the middle of the channel's layer (emission source) or of the second layer (usually the buried one)
  const chLayer = ch ? Math.max(0, info!.layer_names.indexOf(ch.split('|')[0])) : Math.min(1, (g?.L ?? 1) - 1)
  const st0 = g ? stackAt(g.shape, 0, 0) : null
  const yDefault = st0 ? st0.bottom[chLayer] + 0.5 * st0.t[chLayer] : 0
  const yc = g ? Math.min(g.ymax, Math.max(0, yMm ?? yDefault)) : 0
  const zc = g ? g.z0 + zFrac * g.Lz : 0
  const exag = g ? exagUser ?? Math.min(12, Math.max(1, (0.34 * g.Lx) / g.ymax)) : 1
  const unit = info?.unit ?? ''

  return (
    <div className="lf">
      <div className="lf-bar">
        {objects.length > 1 && <div className="f"><span>Object</span><div style={{ width: 130 }}><Sel value={obj ?? ''} options={objects.map((o) => ({ value: o, label: o }))} onChange={setObject} /></div></div>}
        <div className="f"><span>Quantity</span><Seg small value={k} onChange={setKind} options={kinds.map((q) => ({ value: q, label: KIND_LABEL[q], title: KIND_TIP[q] }))} /></div>
        <div className="f"><span>Excitation</span><div className="tabs" style={{ flex: 'none' }}>{exs.map((e) => <button key={e} className={`tab${e === ex ? ' on' : ''}`} style={{ ['--c' as string]: wavelengthToHex(e) }} onClick={() => setEx(e)}><i />{e}</button>)}</div></div>
        {k === 'source' && info && info.channels.length > 0 && <div className="f"><span>Channel</span><div style={{ width: 170 }}><Sel value={ch ?? ''} options={info.channels.map((c) => ({ value: c, label: c.replace('|', ' · ') }))} onChange={setChannel} /></div></div>}
        <span style={{ flex: 1 }} />
        <Busy on={vol.loading} />
      </div>
      {sc && <div className="cbar-h" style={{ padding: '6px 14px', borderBottom: '1px solid var(--line)', background: 'var(--bg1)', flexWrap: 'nowrap', whiteSpace: 'nowrap' }}>
        <span>{fmtVal(sc.vmin)}</span><span className="g" style={{ background: cssGradient(cmap), flex: 'none' }} /><span>{fmtVal(sc.vmax)}</span>
        <span style={{ color: 'var(--accent)', overflow: 'hidden', textOverflow: 'ellipsis' }}>{unit.replace('^2', '²')}{log ? ` · log ${decades} dec` : ' · linear'}</span>
        <span style={{ marginLeft: 'auto', display: 'inline-flex', gap: 8, alignItems: 'center', flex: 'none' }}>
          <Toggle value={log} label="log" onChange={setLog} tip="Logarithmic colour scale — light decays exponentially with depth, so log reveals the buried layers." />
          {log && <div style={{ width: 84 }}><Num value={decades} min={1} max={8} step={1} integer unit="dec" bar={false} onChange={setDecades} /></div>}
          <div style={{ width: 92 }}><Sel value={cmap} options={['inferno', 'magma', 'viridis', 'turbo'] as CmapName[]} onChange={setCmap} /></div>
        </span>
      </div>}

      {!shape && <div className="note warn" style={{ margin: '12px 14px' }}>Layer geometry for this capture is unavailable (its experiment no longer matches the open one), so the light field cannot be drawn in physical coordinates.</div>}
      {vol.error && <div className="note err" style={{ margin: '12px 14px' }}>{vol.error}</div>}

      {g && data && sc && (
        <div className="lf-main">
          <XzCard g={g} data={data} sc={sc} z={zc} zFrac={zFrac} setZFrac={setZFrac} exag={exag} setExag={setExag} autoExag={exagUser == null} y={yc} colors={layerColors} names={info!.layer_names} unit={unit} />
          <div className="lf-grid">
            <XyCard g={g} data={data} sc={sc} y={yc} setY={setY} z={zc} colors={layerColors} names={info!.layer_names} unit={unit} />
            <DepthCard profiles={profiles} names={info!.layer_names} colors={layerColors} unit={unit} log={log} decades={decades} onExport={() => downloadText(`${safeName(`${meta.name}_${info!.object_id}_${k}`)}_depth.json`, toJson('light_field_depth_profile', { captureId: meta.id, experimentName: meta.name, norm: { mode: 'adu' }, extra: { quantity: k, excitation_nm: ex ?? 0, unit } }, profiles.map((p) => ({ layer: info!.layer_names[p.layer], depth_mm: p.depth, mean: p.mean, std: p.std }))), 'application/json')} />
          </div>
        </div>)}
    </div>
  )
}

/* ---------------------------------------------------------------- XZ slice */
function XzCard({ g, data, sc, z, zFrac, setZFrac, exag, setExag, autoExag, y, colors, names, unit }: {
  g: VolGeom; data: Float32Array; sc: ReturnType<typeof autoScale>; z: number; zFrac: number; setZFrac: (v: number) => void
  exag: number; setExag: (v: number | null) => void; autoExag: boolean; y: number; colors: string[]; names: string[]; unit: string
}) {
  const [wrap, w] = useWidth<HTMLDivElement>()
  const cv = useRef<HTMLCanvasElement>(null)
  const dpr = useDpr()
  const [hov, setHov] = useState<{ px: number; py: number; text: string[] } | null>(null)
  const hCss = Math.round(Math.min(380, Math.max(90, g.ymax * exag * (w / g.Lx))))
  useEffect(() => {
    const c = cv.current; if (!c) return
    const W = Math.round(w * dpr), H = Math.round(hCss * dpr)
    c.width = W; c.height = H
    c.getContext('2d')!.putImageData(renderXZ(g, data, z, W, H, sc), 0, 0)
  }, [g, data, z, w, hCss, dpr, sc])

  const Y = (yy: number) => hCss * (1 - yy / g.ymax)
  const stL = stackAt(g.shape, g.x0, z), stR = stackAt(g.shape, g.x1, z)
  const ifaces = Array.from({ length: g.L }, (_, i) => ({ l: stL.bottom[i] + stL.t[i], r: stR.bottom[i] + stR.t[i] }))
  const onMove = (e: React.MouseEvent) => {
    const r = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - r.left, py = e.clientY - r.top
    const x = g.x0 + (px / r.width) * g.Lx, yy = g.ymax * (1 - py / r.height)
    const c = locate(g, x, yy, z)
    if (!c) { setHov(null); return }
    const v = data[(c.ix * g.nz + c.iz) * g.nd + c.d]
    setHov({ px, py, text: [`x ${x.toFixed(1)} mm · y ${yy.toFixed(2)} mm`, `${names[c.layer]}`, `${fmtVal(v)} ${unit}`] })
  }
  return (
    <div className="lf-card">
      <header>XZ section <small>depth versus x at z = {z.toFixed(1)} mm · physical coordinates, vertical ×{exag.toFixed(1)}</small><span className="sp" />
        <div className="f" style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontWeight: 400, color: 'var(--faint)' }}>
          <span>z</span><input type="range" min={0} max={1} step={0.005} value={zFrac} onChange={(e) => setZFrac(+e.target.value)} style={{ width: 140 }} aria-label="z position" />
          <span>stretch</span><div style={{ width: 78 }}><Num value={+exag.toFixed(1)} min={1} max={20} step={0.5} unit="×" bar={false} onChange={(v) => setExag(v)} /></div>
          {!autoExag && <button className="btn ghost sm" onClick={() => setExag(null)}>auto</button>}
        </div>
      </header>
      <div className="cbody">
        <div ref={wrap} className="lf-canvas-wrap" style={{ height: hCss }} onMouseMove={onMove} onMouseLeave={() => setHov(null)}>
          <canvas ref={cv} style={{ width: w, height: hCss }} />
          <svg width={w} height={hCss} style={{ overflow: 'visible' }}>
            {ifaces.map((f, i) => <line key={i} x1={0} y1={Y(f.l)} x2={w} y2={Y(f.r)} stroke={i === g.L - 1 ? '#ffffff' : 'rgba(255,255,255,.7)'} strokeWidth={i === g.L - 1 ? 1.4 : 1} strokeDasharray={i === g.L - 1 ? undefined : '4 3'} />)}
            <line x1={0} x2={w} y1={Y(y)} y2={Y(y)} stroke="#35d8ff" strokeWidth={1} strokeDasharray="2 4" opacity={0.9} />
            <text x={w - 4} y={Y(y) - 3} fontSize="9.5" textAnchor="end" fill="#35d8ff" fontFamily="var(--mono)">XY plane y = {y.toFixed(2)} mm</text>
            {names.map((n, i) => {
              const tl = stL.t[i], tr = stR.t[i]
              const left = tl >= tr
              const th = Math.max(tl, tr) * (hCss / g.ymax)
              if (th < 12) return null
              const yy = left ? Y(stL.bottom[i] + tl / 2) : Y(stR.bottom[i] + tr / 2)
              return <g key={i}><rect x={left ? 8 : w - 8 - n.length * 6.3 - 14} y={yy - 8} width={n.length * 6.3 + 14} height={16} rx={8} fill="rgba(8,10,14,.72)" /><circle cx={left ? 16 : w - 8 - n.length * 6.3 - 6} cy={yy} r={3.5} fill={colors[i]} />
                <text x={left ? 24 : w - 8 - n.length * 6.3 + 2} y={yy + 3.5} fontSize="10.5" fill="#fff" fontWeight="600">{n}</text></g>
            })}
          </svg>
          {hov && <div className="plot-tip" style={{ left: Math.min(hov.px + 14, w - 190), top: Math.max(2, hov.py - 56) }}><b>{hov.text[1]}</b><div>{hov.text[0]}</div><div><span style={{ marginLeft: 0, color: '#fff' }}>{hov.text[2]}</span></div></div>}
        </div>
        <div className="lf-legend">{names.map((n, i) => <span key={i}><i style={{ background: colors[i] }} />{n}</span>)}<span style={{ marginLeft: 'auto', color: 'var(--faint)' }}>dashed = layer interfaces · x from {g.x0.toFixed(0)} to {g.x1.toFixed(0)} mm</span></div>
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------- XY plan */
function XyCard({ g, data, sc, y, setY, z, colors, names, unit }: {
  g: VolGeom; data: Float32Array; sc: ReturnType<typeof autoScale>; y: number; setY: (v: number) => void; z: number; colors: string[]; names: string[]; unit: string
}) {
  const [wrap, w] = useWidth<HTMLDivElement>()
  const cv = useRef<HTMLCanvasElement>(null)
  const dpr = useDpr()
  const hCss = Math.round(Math.min(420, w * (g.Lz / g.Lx)))
  const [hov, setHov] = useState<{ px: number; py: number; text: string[] } | null>(null)
  useEffect(() => {
    const c = cv.current; if (!c) return
    const W = Math.round(Math.min(560, w * dpr)), H = Math.round(Math.min(560, hCss * dpr))
    c.width = W; c.height = H
    c.getContext('2d')!.putImageData(renderXY(g, data, y, W, H, sc), 0, 0)
  }, [g, data, y, w, hCss, dpr, sc])
  const st0 = stackAt(g.shape, 0, 0)
  const onMove = (e: React.MouseEvent) => {
    const r = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - r.left, py = e.clientY - r.top
    const x = g.x0 + (px / r.width) * g.Lx, zz = g.z0 + (py / r.height) * g.Lz
    const c = locate(g, x, y, zz)
    if (!c) { setHov(null); return }
    setHov({ px, py, text: [`x ${x.toFixed(1)} · z ${zz.toFixed(1)} mm`, names[c.layer], `${fmtVal(data[(c.ix * g.nz + c.iz) * g.nd + c.d])} ${unit}`] })
  }
  return (
    <div className="lf-card">
      <header>XY plane <small>plan view at height y = {y.toFixed(2)} mm above the base</small></header>
      <div className="cbody">
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', paddingBottom: 8 }}>
          <input type="range" min={0} max={g.ymax} step={g.ymax / 400} value={y} onChange={(e) => setY(+e.target.value)} aria-label="plane height" />
          <span className="mono" style={{ fontSize: 11, color: 'var(--dim)', minWidth: 54, textAlign: 'right' }}>{y.toFixed(2)} mm</span>
        </div>
        <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', paddingBottom: 8 }}>
          {names.map((n, i) => <button key={n} className="chip2" style={{ ['--c' as string]: colors[i] }} title={`Jump to the middle of ${n} at the footprint centre`} onClick={() => setY(st0.bottom[i] + st0.t[i] / 2)}><i />{n}</button>)}
        </div>
        <div ref={wrap} className="lf-canvas-wrap" style={{ height: hCss }} onMouseMove={onMove} onMouseLeave={() => setHov(null)}>
          <canvas ref={cv} style={{ width: w, height: hCss }} />
          <svg width={w} height={hCss}>
            <line x1={0} x2={w} y1={((z - g.z0) / g.Lz) * hCss} y2={((z - g.z0) / g.Lz) * hCss} stroke="#35d8ff" strokeWidth={1} strokeDasharray="2 4" opacity={0.9} />
            <text x={4} y={((z - g.z0) / g.Lz) * hCss - 3} fontSize="9.5" fill="#35d8ff" fontFamily="var(--mono)">XZ section</text>
          </svg>
          {hov && <div className="plot-tip" style={{ left: Math.min(hov.px + 14, w - 190), top: Math.max(2, hov.py - 56) }}><b>{hov.text[1]}</b><div>{hov.text[0]}</div><div><span style={{ marginLeft: 0, color: '#fff' }}>{hov.text[2]}</span></div></div>}
        </div>
        <div className="lf-legend"><span style={{ color: 'var(--faint)' }}>light edges = layer boundaries cut by the plane</span></div>
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------- depth profile */
function DepthCard({ profiles, names, colors, unit, log, decades, onExport }: { profiles: ReturnType<typeof depthProfiles>; names: string[]; colors: string[]; unit: string; log: boolean; decades: number; onExport: () => void }) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const h = 330
  const m = { l: 44, r: 12, t: 22, b: 30 }
  const iw = Math.max(40, w - m.l - m.r), ih = h - m.t - m.b
  const [hov, setHov] = useState<{ py: number; px: number } | null>(null)
  const depthMax = Math.max(0.1, ...profiles.map((p) => p.bottom))
  const pos = profiles.flatMap((p) => p.mean.filter((v) => v > 0))
  const vmax = pos.length ? Math.max(...pos) : 1
  const vmin = log ? vmax * 10 ** -decades : 0
  const xs = (v: number) => m.l + (log ? Math.log10(Math.max(v, vmin) / vmin) / Math.log10(vmax / vmin || 10) : (v - vmin) / (vmax - vmin || 1)) * iw
  const ys = (d: number) => m.t + (d / depthMax) * ih
  const xTicks = log ? Array.from({ length: decades + 1 }, (_, i) => vmin * 10 ** i) : niceTicks(0, vmax, 4)
  const yTicks = niceTicks(0, depthMax, 5)
  const readout = hov && (() => {
    const d = ((hov.py - m.t) / ih) * depthMax
    let best: { p: (typeof profiles)[number]; s: number; dd: number } | null = null
    for (const p of profiles) p.depth.forEach((dep, s) => { const dd = Math.abs(dep - d); if (!best || dd < best.dd) best = { p, s, dd } })
    return best as { p: (typeof profiles)[number]; s: number; dd: number } | null
  })()
  return (
    <div className="lf-card">
      <header>Depth profile <small>lateral mean ± std</small><span className="sp" /><button className="btn sm" onClick={onExport}><FileJson size={13} />JSON</button></header>
      <div className="cbody">
        <div ref={ref} style={{ position: 'relative' }}>
          <svg width={w} height={h} style={{ display: 'block' }} onMouseMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); setHov({ px: e.clientX - r.left, py: e.clientY - r.top }) }} onMouseLeave={() => setHov(null)}>
            {profiles.map((p) => <rect key={`b${p.layer}`} x={m.l} y={ys(p.top)} width={iw} height={Math.max(0, ys(p.bottom) - ys(p.top))} fill={colors[p.layer]} opacity={0.09} />)}
            {xTicks.map((v, i) => <g key={i}><line x1={xs(v)} x2={xs(v)} y1={m.t} y2={m.t + ih} stroke="rgba(255,255,255,.06)" /><text x={xs(v)} y={h - 14} textAnchor="middle" fontSize="9.5" fill="#6b7788" fontFamily="var(--mono)">{fmtTick(v)}</text></g>)}
            {yTicks.map((v) => <g key={v}><line x1={m.l} x2={m.l + iw} y1={ys(v)} y2={ys(v)} stroke="rgba(255,255,255,.06)" /><text x={m.l - 6} y={ys(v) + 3} textAnchor="end" fontSize="9.5" fill="#6b7788" fontFamily="var(--mono)">{fmtTick(v)}</text></g>)}
            <line x1={m.l} x2={m.l} y1={m.t} y2={m.t + ih} stroke="#3a4350" />
            {profiles.map((p) => {
              const pts = p.depth.map((d, s) => [xs(p.mean[s]), ys(d)] as const)
              const up = p.depth.map((d, s) => `${xs(p.mean[s] + p.std[s]).toFixed(1)},${ys(d).toFixed(1)}`)
              const lo = p.depth.map((d, s) => `${xs(Math.max(p.mean[s] - p.std[s], 0)).toFixed(1)},${ys(d).toFixed(1)}`).reverse()
              return <g key={p.layer}>
                <path d={`M${up.join('L')}L${lo.join('L')}Z`} fill={colors[p.layer]} opacity={0.2} />
                <path d={pts.map((q, i) => `${i ? 'L' : 'M'}${q[0].toFixed(1)},${q[1].toFixed(1)}`).join('')} fill="none" stroke={colors[p.layer]} strokeWidth={2.2} strokeLinejoin="round" />
                {pts.map((q, i) => <circle key={i} cx={q[0]} cy={q[1]} r={2.2} fill="#0b0e12" stroke={colors[p.layer]} strokeWidth={1.4} />)}
              </g>
            })}
            <text x={m.l + iw} y={h - 1} textAnchor="end" fontSize="9.5" fill="#566174">{unit}</text>
            <text x={4} y={12} fontSize="9.5" fill="#566174">depth below top surface (mm)</text>
            {readout && <line x1={m.l} x2={m.l + iw} y1={ys(readout.p.depth[readout.s])} y2={ys(readout.p.depth[readout.s])} stroke="rgba(255,255,255,.3)" />}
          </svg>
          {readout && hov && <div className="plot-tip" style={{ left: Math.min(hov.px + 12, w - 170), top: Math.max(0, hov.py - 54) }}>
            <b>{names[readout.p.layer]}</b><div>depth {readout.p.depth[readout.s].toFixed(2)} mm</div><div><span style={{ marginLeft: 0, color: '#fff' }}>{fmtVal(readout.p.mean[readout.s])} ± {fmtVal(readout.p.std[readout.s])}</span></div></div>}
        </div>
        <div className="lf-legend">{profiles.map((p) => <span key={p.layer}><i style={{ background: colors[p.layer] }} />{names[p.layer]}</span>)}<span style={{ marginLeft: 'auto', color: 'var(--faint)' }}>depth below the top surface</span></div>
      </div>
    </div>
  )
}

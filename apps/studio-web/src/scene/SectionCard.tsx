import { FlipHorizontal2, Slice, Sparkles, X } from 'lucide-react'
import { useStudio } from '../state/store'
import { useLayerView, DEFAULT_SECTION, targetBlockId } from '../state/layerView'
import { cssGradient, type CmapName } from '../ui/colormaps'
import { defaultChannel } from './lightfield'
import { Num, Seg, Sel, Tip } from '../ui/controls'
import { wedgeAngleDeg } from '../three/layers/profile'
import type { BlockLayer } from '../generated/experiment'

/** Floating cutaway controls shown while a layered block's section view is on. */
export function SectionCard() {
  const exp = useStudio((s) => s.exp)
  const selection = useStudio((s) => s.selection)
  const id = targetBlockId(exp, selection)
  const sec = useLayerView((s) => (id ? s.sections[id] ?? DEFAULT_SECTION : DEFAULT_SECTION))
  const lfOn = useLayerView((s) => s.lf.on)
  const setSection = useLayerView((s) => s.setSection)
  const setLf = useLayerView((s) => s.setLf)
  const obj = id ? exp.objects.find((o) => o.id === id) : null
  if (!id || !obj || obj.shape.kind !== 'layered_block' || !sec.on) return null
  const sh = obj.shape
  const half = (sec.axis === 'x' ? sh.footprint_mm[0] : sh.footprint_mm[1]) / 2
  const wedges = (sh.layers as BlockLayer[]).filter((l) => l.thickness.kind === 'linear')
  return (
    <div className="sec-card glass" onPointerDown={(e) => e.stopPropagation()}>
      <span className="ttl"><Slice size={14} />Cutaway</span>
      <Seg small value={sec.axis} onChange={(a) => setSection(id, { axis: a, position_mm: 0 })} options={[{ value: 'x', label: '⟂ x' }, { value: 'z', label: '⟂ z' }]} />
      <div className="grow"><Num value={sec.position_mm} min={-half} max={half} step={0.25} unit="mm" onChange={(v) => setSection(id, { position_mm: v })} /></div>
      <Tip text="Flip the removed side"><button className="btn ghost icon sm" onClick={() => setSection(id, { keep: sec.keep === 'low' ? 'high' : 'low' })}><FlipHorizontal2 size={14} /></button></Tip>
      <span className="sep" />
      <Tip text="Paint the 3D light field (fluence) on the cut plane"><button className={`btn sm${lfOn ? ' on' : ''}`} onClick={() => setLf({ on: !lfOn })}><Sparkles size={13} />Light field</button></Tip>
      {wedges.length > 0 && <span className="hint mono" style={{ whiteSpace: 'nowrap' }}>{wedges.map((w) => `${w.name}: ${wedgeAngleDeg(w, sh.footprint_mm)?.toFixed(1)}° wedge`).join(' · ')}</span>}
      <Tip text="Close cutaway [X]"><button className="btn ghost icon sm" onClick={() => setSection(id, { on: false })}><X size={14} /></button></Tip>
    </div>
  )
}

/** Colour-scale legend of the displayed light field, doubling as its quick controls (quantity + excitation). */
export function LightFieldLegend() {
  const lf = useLayerView((s) => s.lf)
  const setLf = useLayerView((s) => s.setLf)
  const legend = useLayerView((s) => s.lfLegend)
  const status = useLayerView((s) => s.lfStatus)
  const infos = useLayerView((s) => s.lfInfos)
  const info = infos.find((v) => v.kind === lf.kind) ?? infos[0]
  if (!lf.on) return null
  const exs = info?.excitations_nm ?? []
  const fmt = (v: number) => (v === 0 ? '0' : v.toExponential(1).replace('e+', 'e'))
  return (
    <div className="wl-legend glass" onPointerDown={(e) => e.stopPropagation()}>
      <div className="ttl">
        <Seg small value={lf.kind} onChange={(k) => setLf({ kind: k })} options={[{ value: 'fluence', label: 'Fluence' }, { value: 'absorbed', label: 'Absorbed' }, { value: 'source', label: 'Source' }]} />
        {exs.length > 0 && <div style={{ width: 78 }}><Sel value={lf.ex != null && exs.includes(lf.ex) ? lf.ex : exs[0]} options={exs.map((x) => ({ value: x, label: `${x} nm` }))} onChange={(v) => setLf({ ex: v })} /></div>}
      </div>
      {lf.kind === 'source' && (info?.channels.length ?? 0) > 0 && <Sel value={defaultChannel(info!.channels, lf.channel) ?? ''} options={info!.channels.map((c) => ({ value: c, label: c.replace('|', ' · ') }))} onChange={(v) => setLf({ channel: v })} />}
      {legend ? <>
        <div className="wl-bar" style={{ background: cssGradient(legend.cmap as CmapName) }} />
        <div className="ends"><span>{fmt(legend.vmin)}</span><span className="u">{legend.unit} · log</span><span>{fmt(legend.vmax)}</span></div>
        {legend.demo && <div className="demo">Illustrative demo field — not an engine result</div>}
      </> : <div className="hint">{status.state === 'loading' ? 'Loading volume…' : status.message ?? 'No light field available.'}</div>}
    </div>
  )
}

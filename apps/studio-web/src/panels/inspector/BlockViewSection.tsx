import { FlipHorizontal2, Slice, Sparkles } from 'lucide-react'
import { useLayerView, DEFAULT_SECTION, type LightKind } from '../../state/layerView'
import { useStudio } from '../../state/store'
import { setCutaway } from '../../scene/blockView'
import { defaultChannel } from '../../scene/lightfield'
import { cssGradient } from '../../ui/colormaps'
import { Num, Row, Section, Seg, Sel, Toggle } from '../../ui/controls'
import type { LayeredBlockShape } from '../../generated/experiment'

const LF_CMAPS = ['inferno', 'magma', 'turbo', 'viridis'] as const

/** Inspector section "Cutaway & light field" for one layered block (view-only; not saved in the experiment). */
export function BlockViewSection({ objectId, shape }: { objectId: string; shape: LayeredBlockShape }) {
  const sec = useLayerView((s) => s.sections[objectId] ?? DEFAULT_SECTION)
  const setSection = useLayerView((s) => s.setSection)
  const lf = useLayerView((s) => s.lf)
  const setLf = useLayerView((s) => s.setLf)
  const infos = useLayerView((s) => s.lfInfos)
  const status = useLayerView((s) => s.lfStatus)
  const engine = useStudio((s) => s.engine)
  const hasCapture = useStudio((s) => !!s.activeCaptureId)
  const half = (sec.axis === 'x' ? shape.footprint_mm[0] : shape.footprint_mm[1]) / 2
  const info = infos.find((v) => v.kind === lf.kind) ?? infos[0]
  const exs = info?.excitations_nm ?? []
  const why = engine === 'mock' ? 'Needs the Python engine (3D Monte Carlo).' : !hasCapture ? 'Run a capture first.' : null
  return (
    <Section title="Cutaway & light field" icon={<Slice size={13} />} extra={sec.on ? <span className="tag">on</span> : undefined}>
      <Row label="Cutaway" tip="Remove half of the block with an exact planar cut: the layer stack and any wedge show in cross-section on a capped face, with thickness callouts. [X]">
        <Toggle value={sec.on} onChange={(v) => { const o = useStudio.getState().exp.objects.find((x) => x.id === objectId); if (o) setCutaway(o, v, { position_mm: Math.min(Math.max(sec.position_mm, -half), half) }) }} />
      </Row>
      <Row label="Cut plane" tip="Plane normal to x or z.">
        <Seg small value={sec.axis} onChange={(a) => setSection(objectId, { axis: a, position_mm: 0 })} options={[{ value: 'x', label: '⟂ x' }, { value: 'z', label: '⟂ z' }]} />
      </Row>
      <Row label="Position" tip="Local coordinate of the cut plane (block centre = 0).">
        <Num value={sec.position_mm} min={-half} max={half} step={0.25} unit="mm" onChange={(v) => setSection(objectId, { position_mm: v, on: true })} />
      </Row>
      <Row label="">
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <button className="btn sm" onClick={() => setSection(objectId, { keep: sec.keep === 'low' ? 'high' : 'low', on: true })}><FlipHorizontal2 size={13} />Flip side</button>
          <Toggle value={sec.labels} onChange={(v) => setSection(objectId, { labels: v })} label="Labels" />
        </div>
      </Row>
      <div className="lf-block">
        <Row label={<><Sparkles size={12} />Light field</>} tip="Paint the 3D Monte-Carlo fluence / absorbed power / fluorescence source on the cut plane, log-scaled — see scattering and absorption of the cover acting on the buried fluorophore.">
          <Toggle value={lf.on} disabled={!!why && !infos.length && !import.meta.env.DEV} onChange={(v) => { setLf({ on: v }); if (v) { const o = useStudio.getState().exp.objects.find((x) => x.id === objectId); if (o) setCutaway(o, true) } }} />
        </Row>
        {why && <div className="hint">{why}</div>}
        {lf.on && !why && status.state !== 'ready' && status.message && <div className="hint" style={{ color: status.state === 'error' ? 'var(--err)' : undefined }}>{status.message}</div>}
        {lf.on && <>
          <Row label="Quantity" tip="Fluence = photons per area (excitation light everywhere). Absorbed = power deposited per volume. Source = where fluorescence is born.">
            <Seg small value={lf.kind} onChange={(k: LightKind) => setLf({ kind: k })} options={[{ value: 'fluence', label: 'Fluence' }, { value: 'absorbed', label: 'Absorbed' }, { value: 'source', label: 'Source' }]} />
          </Row>
          {exs.length > 0 && <Row label="Excitation"><Sel value={lf.ex != null && exs.includes(lf.ex) ? lf.ex : exs[0]} options={exs.map((x) => ({ value: x, label: `${x} nm` }))} onChange={(v) => setLf({ ex: v })} /></Row>}
          {lf.kind === 'source' && (info?.channels.length ?? 0) > 0 && <Row label="Channel"><Sel value={defaultChannel(info!.channels, lf.channel) ?? ''} options={info!.channels.map((c) => ({ value: c, label: c.replace('|', ' · ') }))} onChange={(v) => setLf({ channel: v })} /></Row>}
          <Row label="Range" tip="Dynamic range of the log colour scale, in decades below the maximum."><Num value={lf.decades} min={1} max={9} step={0.5} unit="dec" onChange={(v) => setLf({ decades: v })} /></Row>
          <Row label="Colours">
            <div className="cmap-pick">{LF_CMAPS.map((c) => (
              <button key={c} className={lf.cmap === c ? 'on' : ''} title={c} onClick={() => setLf({ cmap: c })} style={{ background: cssGradient(c) }} />
            ))}</div>
          </Row>
        </>}
      </div>
    </Section>
  )
}

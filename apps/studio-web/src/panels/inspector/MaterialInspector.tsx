import { useMemo, useState } from 'react'
import { Plus, X } from 'lucide-react'
import { useStudio, activeSource } from '../../state/store'
import { BUNDLE, fluorophoreMap } from '../../state/presets'
import { Section, Row, Sel, Num, Seg, Tip } from '../../ui/controls'
import { LineChart, Heatmap } from '../../ui/charts'
import { InspectorHeader, HeaderActions, NF, TF, SF, BF } from './Fields'
import { wavelengthToHex } from '../../lib/color'
import { eem, absorberMua, materialMua, muaDye, musPrime, range } from '../../lib/spectral'
import type { MaterialSpec, FluorophoreSpec } from '../../generated/experiment'
import { FlaskConical } from '../../ui/icons'

export const dyeColor = (f?: FluorophoreSpec) => f?.display_color ?? wavelengthToHex(f?.em_peak_nm ?? 550)

export function MaterialInspector({ id }: { id: string }) {
  const mat = useStudio((s) => s.exp.materials.find((m) => m.id === id))
  const exp = useStudio((s) => s.exp)
  const edit = useStudio((s) => s.edit)
  const dupe = useStudio((s) => s.duplicateSelected), del = useStudio((s) => s.deleteSelected)
  const [logY, setLogY] = useState(true)
  const fl = useMemo(() => fluorophoreMap(exp), [exp])
  const src = activeSource(exp)
  const exs = src.sequence.filter((s) => s.enabled).map((s) => s.wavelength_nm)
  const wl = useMemo(() => range(330, 800, 5), [])
  const eemEx = useMemo(() => range(300, 700, 10), []), eemEm = useMemo(() => range(350, 800, 10), [])
  const eemData = useMemo(() => (mat ? eem(mat, fl, eemEx, eemEm) : []), [mat, fl, eemEx, eemEm])
  if (!mat) return null
  const upd = (fn: (m: MaterialSpec) => void, key?: string) => edit((e) => { const m = e.materials.find((x) => x.id === id); if (m) fn(m) }, key ?? `mat:${id}`)
  const used = exp.objects.some((o) => o.material_id === id || o.vessel_material_id === id) || exp.background.material_id === id
  const allDyes = [...new Set([...BUNDLE.fluorophores, ...exp.fluorophores].map((f) => f.name))]
  const dyeNames = mat.dyes.map((d) => d.fluorophore)

  const muaTotal = wl.map((w) => materialMua(mat, fl, w))
  const series = [
    ...mat.dyes.map((d) => ({ name: d.fluorophore, x: wl, y: wl.map((w) => (fl[d.fluorophore] ? muaDye(fl[d.fluorophore], w, d.concentration_uM) : 0)), color: dyeColor(fl[d.fluorophore]), width: 1.2, opacity: 0.8, dash: '4 3' })),
    ...(mat.background_absorption.preset !== 'none' ? [{ name: mat.background_absorption.preset, x: wl, y: wl.map((w) => absorberMua(mat.background_absorption, w)), color: '#7d8798', width: 1.2, opacity: 0.8, dash: '2 3' }] : []),
    { name: 'total μa', x: wl, y: muaTotal.map((v) => Math.max(v, 1e-7)), color: '#e8ecf2', fill: true, width: 1.8 },
  ]

  return (
    <>
      <InspectorHeader icon={<FlaskConical size={20} />} title={mat.name} sub={`Material · ${mat.id}${used ? '' : ' · unused'}`}
        actions={<HeaderActions onDup={dupe} onDel={del} delDisabled={used} delTitle={used ? 'In use by an object' : 'Delete'} />} />
      <Section title="Identity">
        <TF label="Name" value={mat.name} onChange={(v) => upd((m) => { m.name = v }, `mat:${id}:name`)} />
        <Row label="Display colour"><div style={{ display: 'flex', gap: 8, alignItems: 'center' }}><input type="color" value={mat.display_color ?? '#888888'} onChange={(e) => upd((m) => { m.display_color = e.target.value }, `mat:${id}:color`)} /><span className="mono hint">{mat.display_color}</span></div></Row>
      </Section>
      <Section title="Fluorophores" extra={
        <Sel value="" options={[{ value: '', label: '+ add dye…' }, ...allDyes.filter((n) => !dyeNames.includes(n)).map((n) => ({ value: n, label: n }))]}
          onChange={(v) => { if (v) upd((m) => { m.dyes.push({ fluorophore: v, concentration_uM: 10 }) }) }} />}>
        {mat.dyes.length === 0 && <div className="hint">No dyes: this material is optically passive (still absorbs/scatters/reflects). Add one above or drag from Library → Dyes.</div>}
        {mat.dyes.map((d, i) => {
          const f = fl[d.fluorophore]
          return (
            <div key={d.fluorophore} style={{ display: 'grid', gap: 5, padding: '8px 9px', background: 'var(--bg2)', border: '1px solid var(--line)', borderRadius: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="swatch" style={{ width: 10, height: 10, borderRadius: 3, background: dyeColor(f) }} />
                <b style={{ flex: 1 }}>{d.fluorophore}</b>
                {f && <span className="tag">{f.ex_peak_nm} → {f.em_peak_nm} nm</span>}
                <button className="btn ghost icon sm" onClick={() => upd((m) => { m.dyes.splice(i, 1); m.fret = m.fret.filter((p) => p.donor !== d.fluorophore && p.acceptor !== d.fluorophore) })}><X size={13} /></button>
              </div>
              <NF label="Concentration" tip="Dye concentration. Absorption μa = ln10·ε·c/10 (ε = extinction × 55 000 M⁻¹cm⁻¹ × excitation profile)." value={d.concentration_uM} min={0} max={5000} step={0.5} unit="µM" onChange={(v) => upd((m) => { m.dyes[i].concentration_uM = v }, `mat:${id}:c${i}`)} />
            </div>
          )
        })}
      </Section>
      <Section title="Optics">
        <NF label="Refractive idx" tip="n of the medium (Fresnel interface, TIR)." value={mat.refractive_index} min={1} max={3} step={0.005} onChange={(v) => upd((m) => { m.refractive_index = v })} />
        <SF label="Scattering" tip="Reduced-scattering power law μs′(λ) = a·(λ/500)^−b with Henyey–Greenstein anisotropy g." value={mat.scattering.model} options={[{ value: 'none', label: 'none (clear)' }, { value: 'power_law', label: 'power law' }]} onChange={(v) => upd((m) => { m.scattering.model = v })} />
        {mat.scattering.model === 'power_law' && <>
          <NF label="μs′ (500 nm)" value={mat.scattering.mus_prime_500_per_mm} min={0} max={50} step={0.05} unit="1/mm" onChange={(v) => upd((m) => { m.scattering.mus_prime_500_per_mm = v }, `mat:${id}:mus`)} />
          <NF label="Power b" value={mat.scattering.power_b} min={0} max={4} step={0.05} onChange={(v) => upd((m) => { m.scattering.power_b = v }, `mat:${id}:b`)} />
          <NF label="Anisotropy g" value={mat.scattering.g} min={-0.99} max={0.99} step={0.01} onChange={(v) => upd((m) => { m.scattering.g = v }, `mat:${id}:g`)} /></>}
        <SF label="Absorber" tip="Non-fluorescent background absorption added to the dyes. Scale: for water 1 = pure water; for haemoglobin/melanin it is μa at 500 nm (1/mm)." value={mat.background_absorption.preset}
          options={['none', 'water', 'hemoglobin', 'melanin', 'carbon_black', 'custom'] as const} onChange={(v) => upd((m) => { m.background_absorption.preset = v })} />
        {mat.background_absorption.preset !== 'none' && <NF label="Absorber scale" value={mat.background_absorption.scale} min={0} max={50} step={0.01} onChange={(v) => upd((m) => { m.background_absorption.scale = v }, `mat:${id}:ascale`)} />}
        {mat.background_absorption.preset === 'custom' && (
          <Row label="μa(λ) table" stack tip="One 'nm, 1/mm' pair per line.">
            <textarea className="field mono" rows={4} defaultValue={mat.background_absorption.custom.map((p) => p.join(', ')).join('\n')}
              onBlur={(e) => upd((m) => { m.background_absorption.custom = e.target.value.split('\n').map((l) => l.split(/[,\s]+/).map(Number)).filter((p) => p.length >= 2 && p.every(Number.isFinite)).map((p) => [p[0], p[1]] as [number, number]) })} />
          </Row>)}
      </Section>
      <Section title="Surface & photophysics" defaultOpen={false}>
        <NF label="Roughness" tip="GGX microfacet roughness for specular glints (0 = mirror-like)." value={mat.surface_roughness} min={0} max={1} step={0.01} onChange={(v) => upd((m) => { m.surface_roughness = v })} />
        <BF label="Aqueous" tip="Water-based medium: adds the 3400 cm⁻¹ Raman band of water." value={mat.aqueous} onChange={(v) => upd((m) => { m.aqueous = v })} />
        <NF label="Raman strength" value={mat.raman_strength} min={0} max={10} step={0.1} onChange={(v) => upd((m) => { m.raman_strength = v })} />
        <NF label="Quenching Ksv" tip="Stern–Volmer self-quenching: F/F0 = 1/(1 + Ksv·c)." value={mat.quenching_ksv_per_uM} min={0} max={1} step={0.0005} precision={4} unit="/µM" onChange={(v) => upd((m) => { m.quenching_ksv_per_uM = v })} />
      </Section>
      <Section title="FRET pairs" defaultOpen={mat.fret.length > 0} extra={mat.dyes.length >= 2 && <button className="btn sm" onClick={() => upd((m) => { m.fret.push({ donor: m.dyes[0].fluorophore, acceptor: m.dyes[1].fluorophore, k_per_uM: 0.05 }) })}><Plus size={12} />Add</button>}>
        {mat.fret.length === 0 && <div className="hint">{mat.dyes.length < 2 ? 'Needs two dyes in the material.' : 'Donor energy transfers to an acceptor with efficiency k·[acceptor]/(1 + k·[acceptor]).'}</div>}
        {mat.fret.map((p, i) => (
          <div key={i} className="grid2" style={{ alignItems: 'center' }}>
            <Sel value={p.donor} options={dyeNames} onChange={(v) => upd((m) => { m.fret[i].donor = v })} />
            <Sel value={p.acceptor} options={dyeNames} onChange={(v) => upd((m) => { m.fret[i].acceptor = v })} />
            <Num value={p.k_per_uM} min={0} max={5} step={0.005} precision={3} unit="/µM" onChange={(v) => upd((m) => { m.fret[i].k_per_uM = v }, `mat:${id}:fret${i}`)} />
            <button className="btn ghost sm" onClick={() => upd((m) => { m.fret.splice(i, 1) })}>Remove</button>
          </div>))}
      </Section>
      <Section title="Spectral view" extra={<Seg small value={logY ? 'log' : 'lin'} onChange={(v) => setLogY(v === 'log')} options={[{ value: 'log', label: 'log' }, { value: 'lin', label: 'lin' }]} />}>
        <div className="hint" style={{ marginTop: -2 }}>Absorption μa(λ) (1/mm) — dashed lines show each component; coloured ticks mark the sequence excitations.</div>
        <LineChart series={series} height={140} xDomain={[330, 800]} logY={logY} yDomain={logY ? [-5, 1.5] : undefined} strip xLabel="nm" yLabel="μa 1/mm"
          markers={exs.map((e) => ({ x: e, color: wavelengthToHex(e) }))} legend />
        {mat.scattering.model === 'power_law' && <>
          <div className="hint">Reduced scattering μs′(λ) (1/mm)</div>
          <LineChart series={[{ x: wl, y: wl.map((w) => musPrime(mat.scattering, w)), color: '#9bb4ff', fill: true }]} height={78} xDomain={[330, 800]} strip xLabel="nm" /></>}
        {mat.dyes.length > 0 && <>
          <Tip text="Excitation–emission matrix (relative, absorption × Φ × emission shape; dilute limit). Hot spots are what the sequence can reveal."><div className="hint" style={{ marginTop: 4 }}>Excitation–emission matrix</div></Tip>
          <Heatmap data={eemData} xs={eemEm} ys={eemEx} height={150} xLabel="emission nm" yLabel="excitation nm" marks={exs.map((e) => ({ y: e, color: wavelengthToHex(e) }))} />
        </>}
      </Section>
    </>
  )
}

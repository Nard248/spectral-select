import { useMemo } from 'react'
import { ArrowUp, ArrowDown, Plus, Trash2 } from 'lucide-react'
import { useStudio } from '../../state/store'
import { fluorophoreMap } from '../../state/presets'
import { Section, Row, Num, Toggle, Tip } from '../../ui/controls'
import { LineChart } from '../../ui/charts'
import { InspectorHeader, HeaderActions, NF, TF, VF, SF, BF, useErrAt, errAt } from './Fields'
import { SourceIcon } from '../../ui/icons'
import { gaussian, excitationProfile, range } from '../../lib/spectral'
import { wavelengthToHex, isInvisible } from '../../lib/color'
import { dyeColor } from './MaterialInspector'
import type { LightSourceSpec, ExcitationStep } from '../../generated/experiment'

function StepRow({ srcId, srcIdx, i, st, n }: { srcId: string; srcIdx: number; i: number; st: ExcitationStep; n: number }) {
  const edit = useStudio((s) => s.edit)
  const eWl = useErrAt(['sources', srcIdx, 'sequence', i, 'wavelength_nm'])
  const eEx = useErrAt(['sources', srcIdx, 'sequence', i, 'exposure_ms'])
  const upd = (fn: (s: LightSourceSpec) => void, key?: string) => edit((e) => { const s = e.sources.find((x) => x.id === srcId); if (s) fn(s) }, key)
  const c = wavelengthToHex(st.wavelength_nm)
  return (
    <tr style={{ opacity: st.enabled ? 1 : 0.45 }}>
      <td style={{ width: 34 }}><Toggle value={st.enabled} onChange={(v) => upd((s) => { s.sequence[i].enabled = v })} /></td>
      <td style={{ width: 14 }}><span style={{ display: 'block', width: 8, height: 8, borderRadius: 4, background: c, boxShadow: `0 0 8px ${c}` }} /></td>
      <td><Num value={st.wavelength_nm} min={200.5} max={1099} step={1} bar={false} error={!!eWl} onChange={(v) => upd((s) => { s.sequence[i].wavelength_nm = v }, `seq${srcId}${i}w`)} /></td>
      <td><Num value={st.power_mW} min={0} step={0.1} bar={false} onChange={(v) => upd((s) => { s.sequence[i].power_mW = v }, `seq${srcId}${i}p`)} /></td>
      <td><Num value={st.exposure_ms} min={0.001} step={5} bar={false} error={!!eEx} onChange={(v) => upd((s) => { s.sequence[i].exposure_ms = v }, `seq${srcId}${i}e`)} /></td>
      <td style={{ whiteSpace: 'nowrap' }}>
        <button className="btn ghost icon sm" disabled={i === 0} onClick={() => upd((s) => { const [a] = s.sequence.splice(i, 1); s.sequence.splice(i - 1, 0, a) })}><ArrowUp size={12} /></button>
        <button className="btn ghost icon sm" disabled={i === n - 1} onClick={() => upd((s) => { const [a] = s.sequence.splice(i, 1); s.sequence.splice(i + 1, 0, a) })}><ArrowDown size={12} /></button>
        <button className="btn ghost icon sm" onClick={() => upd((s) => { s.sequence.splice(i, 1) })}><Trash2 size={12} /></button>
      </td>
    </tr>
  )
}

export function SourceInspector({ id }: { id: string }) {
  const idx = useStudio((s) => s.exp.sources.findIndex((x) => x.id === id))
  const src = useStudio((s) => s.exp.sources[idx])
  const exp = useStudio((s) => s.exp)
  const edit = useStudio((s) => s.edit)
  const active = useStudio((s) => s.exp.active_source_id === id)
  const dupe = useStudio((s) => s.duplicateSelected), del = useStudio((s) => s.deleteSelected)
  const errors = useStudio((s) => s.errors)
  const fl = useMemo(() => fluorophoreMap(exp), [exp])
  const wl = useMemo(() => range(280, 700, 2), [])
  if (!src) return null
  const upd = (fn: (s: LightSourceSpec) => void, key?: string) => edit((e) => { const s = e.sources.find((x) => x.id === id); if (s) fn(s) }, key ?? `src:${id}`)
  const dyes = [...new Set(exp.materials.filter((m) => exp.objects.some((o) => o.material_id === m.id || o.vessel_material_id === m.id)).flatMap((m) => m.dyes.map((d) => d.fluorophore)))]
  const lines = src.sequence.filter((s) => s.enabled)
  const series = [
    ...dyes.filter((d) => fl[d]).map((d) => ({ name: `${d} abs.`, x: wl, y: wl.map((w) => excitationProfile(fl[d], w)), color: dyeColor(fl[d]), fill: true, opacity: 0.9, width: 1.3 })),
    ...lines.map((s) => ({ x: wl, y: wl.map((w) => gaussian(w, s.wavelength_nm, src.line_fwhm_nm) * 1.05), color: wavelengthToHex(s.wavelength_nm), width: 1.5 })),
  ]
  const nextWl = lines.length ? Math.min(1090, Math.max(...lines.map((s) => s.wavelength_nm)) + 20) : 405
  const totalS = src.sequence.filter((s) => s.enabled).reduce((a, s) => a + s.exposure_ms, 0) / 1000

  return (
    <>
      <InspectorHeader icon={<SourceIcon kind={src.kind} size={20} />} title={src.name} sub={`Light source · ${src.id}${active ? ' · active' : ''}`}
        actions={<HeaderActions onDup={dupe} onDel={del} delDisabled={exp.sources.length < 2} delTitle="Delete (needs at least one source)" />} />
      <Section title="Source">
        <TF label="Name" value={src.name} onChange={(v) => upd((s) => { s.name = v }, `src:${id}:name`)} />
        <SF label="Type" tip="Tunable lamp: Xe arc × Gaussian monochromator pass. LED: asymmetric emission. Laser: very narrow line." value={src.kind} options={[{ value: 'tunable_lamp', label: 'Tunable lamp (Xe + mono)' }, { value: 'led_array', label: 'LED array' }, { value: 'laser', label: 'Laser' }]} onChange={(v) => upd((s) => { s.kind = v })} />
        <BF label="Active" tip="Only the active source and camera are simulated." value={active} onChange={(v) => { if (v) edit((e) => { e.active_source_id = id }) }} />
      </Section>
      <Section title="Pose">
        <VF label="Position" value={src.position_mm} step={1} unit="mm" onChange={(v) => upd((s) => { s.position_mm = v }, `src:${id}:pos`)} />
        <VF label="Target" tip="Aim point. Drag the ● handle in the viewport (⇧ = vertical)." value={src.target_mm} step={1} unit="mm" onChange={(v) => upd((s) => { s.target_mm = v }, `src:${id}:tgt`)} />
        <NF label="Roll" value={src.roll_deg} min={-180} max={180} step={1} unit="°" onChange={(v) => upd((s) => { s.roll_deg = v }, `src:${id}:roll`)} />
      </Section>
      <Section title="Beam">
        <SF label="Profile" tip="Gaussian: I ∝ exp(−2r²/w²) with w(z) = a/2 + z·tanθ. Flat-top: uniform disc with soft edge." value={src.beam.profile} options={['gaussian', 'flat_top'] as const} onChange={(v) => upd((s) => { s.beam.profile = v })} />
        <NF label="Aperture ⌀" tip="Beam diameter at the exit (1/e² for Gaussian)." value={src.beam.aperture_diameter_mm} min={0.1} max={100} step={0.5} unit="mm" err={errAt(errors, ['sources', idx, 'beam', 'aperture_diameter_mm'])} onChange={(v) => upd((s) => { s.beam.aperture_diameter_mm = v }, `src:${id}:ap`)} />
        <NF label="Divergence" tip="Half-angle of the beam cone." value={src.beam.divergence_half_angle_deg} min={0} max={79} step={0.1} unit="°" onChange={(v) => upd((s) => { s.beam.divergence_half_angle_deg = v }, `src:${id}:div`)} />
      </Section>
      <Section title="Spectrum" defaultOpen={false}>
        <NF label="Line FWHM" tip="Monochromator bandpass / LED width / laser linewidth." value={src.line_fwhm_nm} min={0.1} max={200} step={0.5} unit="nm" onChange={(v) => upd((s) => { s.line_fwhm_nm = v }, `src:${id}:fwhm`)} />
        <SF label="Lamp curve" tip="Xenon arc: UV-rich lines on a continuum; flat: constant spectral power." value={src.lamp_curve} options={['xenon', 'flat'] as const} onChange={(v) => upd((s) => { s.lamp_curve = v })} />
        <NF label="Filter OD" tip="Optical density of the excitation clean-up filter: out-of-band leakage is 10^−OD." value={src.excitation_filter_od} min={0} max={8} step={0.1} onChange={(v) => upd((s) => { s.excitation_filter_od = v }, `src:${id}:od`)} />
        <NF label="Power drift" tip="1σ step-to-step power fluctuation." value={src.power_drift_pct} min={0} max={20} step={0.1} unit="%" onChange={(v) => upd((s) => { s.power_drift_pct = v }, `src:${id}:drift`)} />
      </Section>
      <Section title={<>Excitation sequence <span className="tag">{lines.length}/{src.sequence.length}</span></>}
        extra={<button className="btn sm" onClick={() => upd((s) => { s.sequence.push({ wavelength_nm: nextWl, power_mW: lines.at(-1)?.power_mW ?? 3, exposure_ms: lines.at(-1)?.exposure_ms ?? 100, enabled: true }) })}><Plus size={12} />Add</button>}>
        <div className="hint" style={{ marginTop: -4 }}>Order matters: each step bleaches the sample before the next. Est. exposure {totalS.toFixed(1)} s / band.</div>
        <table className="tbl"><thead><tr><th /><th /><th>λ nm</th><th>mW</th><th>ms</th><th /></tr></thead>
          <tbody>{src.sequence.map((st, i) => <StepRow key={i} srcId={id} srcIdx={idx} i={i} st={st} n={src.sequence.length} />)}</tbody></table>
        {src.sequence.some((s) => s.enabled && isInvisible(s.wavelength_nm)) && <div className="hint">Violet glow marks invisible (UV/NIR) lines in the viewport.</div>}
      </Section>
      <Section title="Lines vs. dye absorption">
        <div className="hint" style={{ marginTop: -4 }}>Sequence lines (Gaussian, FWHM {src.line_fwhm_nm} nm) over the absorption of dyes in the scene.</div>
        <LineChart series={series} height={130} xDomain={[280, 700]} yDomain={[0, 1.12]} strip xLabel="nm" legend />
        <Row label=""><Tip text="Photobleaching depends on the order of the steps."><span className="hint">⚠ high-energy steps first bleach the sample more.</span></Tip></Row>
      </Section>
    </>
  )
}

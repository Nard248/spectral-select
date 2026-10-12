import { useStudio } from '../../state/store'
import { Section, Toggle, Tip, Row, Seg } from '../../ui/controls'
import { InspectorHeader, NF, SF, BF } from './Fields'
import { BackgroundIcon } from '../../ui/icons'
import { Layers, SlidersHorizontal, Info, Box as BoxIcon } from 'lucide-react'
import { estimateVolumeCost, fmtDuration, qualityOf, QUALITY_PRESETS, ASSUMED_PHOTONS_PER_S } from '../../state/volumeEstimate'
import type { PhysicsToggles } from '../../generated/experiment'

export function BackgroundInspector() {
  const bg = useStudio((s) => s.exp.background)
  const materials = useStudio((s) => s.exp.materials)
  const edit = useStudio((s) => s.edit)
  const upd = (fn: (b: typeof bg) => void, key?: string) => edit((e) => fn(e.background), key ?? 'bg')
  return (
    <>
      <InspectorHeader icon={<BackgroundIcon size={20} />} title="Background & room" sub="Table surface and ambient light" />
      <Section title="Table">
        <SF label="Kind" tip="'plane': a slab of the chosen material covering the table (y ∈ [−thickness, 0]). 'none': rays that miss every object return nothing." value={bg.kind} options={[{ value: 'plane', label: 'Plane' }, { value: 'none', label: 'None (void)' }]} onChange={(v) => upd((b) => { b.kind = v })} />
        {bg.kind === 'plane' && <>
          <SF label="Material" value={bg.material_id ?? ''} options={[{ value: '', label: '— none —' }, ...materials.map((m) => ({ value: m.id, label: m.name }))]} onChange={(v) => upd((b) => { b.material_id = v || null })} />
          <NF label="Thickness" value={bg.thickness_mm} min={0.1} max={100} step={0.5} unit="mm" onChange={(v) => upd((b) => { b.thickness_mm = v }, 'bg:t')} />
          <NF label="Extent" value={bg.extent_mm} min={50} max={5000} step={10} unit="mm" onChange={(v) => upd((b) => { b.extent_mm = v }, 'bg:x')} /></>}
      </Section>
      <Section title="Room light">
        <BF label="Enabled" tip="Ambient room light adds a broadband (or line-spectrum) pedestal to every band." value={bg.room_light.enabled} onChange={(v) => upd((b) => { b.room_light.enabled = v })} />
        <SF label="Spectrum" tip="Fluorescent tubes have Hg lines (405, 436, 546, 578 nm) that leak into the bands." value={bg.room_light.spectrum} options={['fluorescent_tube', 'white_led', 'daylight', 'incandescent'] as const} onChange={(v) => upd((b) => { b.room_light.spectrum = v })} />
        <NF label="Illuminance" value={bg.room_light.illuminance_lux} min={0} max={2000} step={1} unit="lux" onChange={(v) => upd((b) => { b.room_light.illuminance_lux = v }, 'bg:lux')} />
      </Section>
    </>
  )
}

type Group = { title: string; items: [keyof PhysicsToggles, string, string][] }
const GROUPS: Group[] = [
  { title: 'Light transport', items: [
    ['fluorescence', 'Fluorescence', 'Dye emission after absorption: Φ · ε · excitation profile × emission shape. Off = only elastic light remains.'],
    ['absorption', 'Absorption', 'Dye + background absorption. Inner-filter and reabsorption emerge from transport — no ad-hoc knobs.'],
    ['scattering', 'Scattering', 'Volume multiple scattering via Monte Carlo transport (μs′ power law, Henyey–Greenstein g).'],
    ['elastic_leak', 'Elastic leak', 'Reflected / backscattered excitation light leaking through the emission filter stack and clean-up filter tails.'],
    ['raman', 'Raman', 'Water Raman (3400 cm⁻¹) in aqueous media: a weak line that follows the excitation, not the dye.'],
    ['fresnel', 'Fresnel', 'Interface reflection, refraction and total internal reflection (unpolarised).'],
    ['specular_glints', 'Specular glints', 'GGX microfacet highlights from the source reflected toward the camera.'],
    ['shadows', 'Shadows', 'Objects shadow each other from the source.'],
    ['subsurface_spread', 'Subsurface spread', 'Lateral light spread under the surface (radial kernel): blurs edges of turbid samples.'],
    ['indirect_bounce', 'Indirect bounce', 'Not yet implemented (v1): the engine does not read this switch. Planned: one diffuse bounce between objects and the table.'],
  ] },
  { title: 'Photophysics', items: [
    ['secondary_emission', 'Secondary emission', 'Re-emission of fluorescence that was reabsorbed by another dye in the same medium.'],
    ['quenching', 'Quenching', 'Stern–Volmer self-quenching F/F0 = 1/(1 + Ksv·c).'],
    ['fret', 'FRET', 'Förster energy transfer from donors to acceptors defined in the material.'],
    ['photobleaching', 'Photobleaching', 'Per-pixel photon dose accumulates in sequence order: c ← c·exp(−Φ_b · absorbed photons per molecule). Order matters.'],
    ['power_drift', 'Power drift', 'Step-to-step fluctuation of lamp power (1σ from the source setting).'],
  ] },
  { title: 'Optics', items: [
    ['defocus', 'Defocus', 'Thin-lens depth of field: blur grows with distance from the focus plane.'],
    ['diffraction', 'Diffraction', 'Airy PSF of the aperture (≈ Gaussian σ = 0.42·λ·N).'],
    ['chromatic_aberration', 'Chromatic aberration', 'Axial focal shift per 100 nm from 550 nm.'],
    ['vignetting', 'Vignetting', 'cos⁴ falloff toward the image corners.'],
    ['filter_angle_shift', 'Filter angle shift', 'Interference filters blue-shift off-axis: λ(θ) = λ₀·√(1 − sin²θ/n_eff²).'],
  ] },
  { title: 'Detector', items: [
    ['shot_noise', 'Shot noise', 'Poisson photon statistics.'],
    ['read_noise', 'Read noise', 'Gaussian electronics noise per readout.'],
    ['dark_current', 'Dark current', 'Thermal signal growing with exposure time.'],
    ['fixed_pattern_noise', 'Fixed pattern', 'PRNU gain map, DSNU offset map and hot pixels.'],
    ['saturation', 'Saturation', 'Pixels clip at the full-well capacity.'],
    ['quantization', 'Quantization', 'ADC rounding at the chosen bit depth, plus the ADU offset.'],
  ] },
]

/** "3D Monte Carlo" group: quality knobs of `acquisition.volume_mc` with a cost estimate. */
function VolumeMcSection() {
  const exp = useStudio((s) => s.exp)
  const edit = useStudio((s) => s.edit)
  const v = exp.acquisition.volume_mc
  const cost = estimateVolumeCost(exp)
  const set = (fn: (m: typeof v) => void, key: string) => edit((e) => fn(e.acquisition.volume_mc), `vmc:${key}`)
  const q = qualityOf(v)
  const hasBlock = cost.blocks > 0
  return (
    <Section title="3D Monte Carlo" icon={<BoxIcon size={13} />} defaultOpen={hasBlock}
      extra={<span className="tag">{hasBlock ? `~${fmtDuration(cost.seconds)}` : 'no block'}</span>}>
      <div className="hint">Quality ↔ time for layered blocks: every excitation step and every emission wavelength node launches photon packets through the 3D layer stack. More photons → smoother deep tails (noise ∝ 1/√N) but run time grows linearly.</div>
      <Row label="Quality" tip="Sets both photon counts. Draft is for composing the scene, Standard for most captures, High/Max for the weak signal under a thick cover.">
        <Seg small value={q as string} onChange={(k) => { const p = QUALITY_PRESETS.find((x) => x.key === k); if (p) set((m) => { m.excitation_photons = p.photons; m.emission_photons = p.photons }, 'q') }}
          options={[...QUALITY_PRESETS.map((p) => ({ value: p.key, label: p.label, title: `${(p.photons / 1e6).toLocaleString()} M photons — ${p.hint}` })), ...(q === 'custom' ? [{ value: 'custom', label: 'Custom' }] : [])]} />
      </Row>
      <NF label="Excitation" tip="Photon packets per excitation step. They carry the beam into the block: sets the noise of the fluence / absorbed-power maps and of the fluorescence source." integer min={10_000} max={200_000_000} step={100_000} value={v.excitation_photons} bar={false} unit="ph" onChange={(x) => set((m) => { m.excitation_photons = x }, 'ex')} />
      <NF label="Emission" tip="Photon packets per emission wavelength node, launched from the fluorescence source volume. Sets the noise of the image of the buried fluorophore." integer min={10_000} max={200_000_000} step={100_000} value={v.emission_photons} bar={false} unit="ph" onChange={(x) => set((m) => { m.emission_photons = x }, 'em')} />
      <NF bar={false} label="Preview" tip="Preview renders use only this fraction of the photons (fast, noisier)." min={0.01} max={1} step={0.01} value={v.preview_fraction} onChange={(x) => set((m) => { m.preview_fraction = x }, 'pf')} />
      <NF bar={false} label="Lateral cell" tip="Tally voxel size in x and z. Smaller = finer maps but fewer photons per voxel (and more memory)." min={0.05} max={5} step={0.05} unit="mm" value={v.lateral_cell_mm} onChange={(x) => set((m) => { m.lateral_cell_mm = x }, 'lc')} />
      <NF bar={false} label="Sublayers" tip="Depth cells across each layer's local thickness (the grid follows the interfaces, so a wedge keeps full depth resolution)." integer min={1} max={64} step={1} value={v.sublayers} onChange={(x) => set((m) => { m.sublayers = x }, 'sl')} />
      <NF bar={false} label="Exit cell" tip="Pixel size of the exit-radiance tally on the faces the camera sees." min={0.05} max={5} step={0.05} unit="mm" value={v.exit_cell_mm} onChange={(x) => set((m) => { m.exit_cell_mm = x }, 'ec')} />
      <NF bar={false} label="Em. step" tip="Spacing of the emission wavelength nodes; the spectrum between nodes is interpolated. Cost ∝ number of nodes." min={2} max={100} step={1} unit="nm" value={v.emission_node_step_nm} onChange={(x) => set((m) => { m.emission_node_step_nm = x }, 'es')} />
      <NF bar={false} label="Acceptance" tip="Half-angle of the camera acceptance cone used for the exit tally (the lens collects light within this angle of the optical axis)." min={0.5} max={60} step={0.5} unit="°" value={v.acceptance_half_angle_deg} onChange={(x) => set((m) => { m.acceptance_half_angle_deg = x }, 'aa')} />
      {hasBlock
        ? <Tip text={`Cost model: (N_ex × ${cost.excitationSteps} steps + N_em × ${cost.emissionNodes} nodes) × ${cost.blocks} block(s) ≈ ${(cost.totalPhotons / 1e6).toFixed(1)} M photon packets at an assumed ~${ASSUMED_PHOTONS_PER_S / 1e6} M/s. Calibrate with your first capture.`}>
          <div className="vmc-est">
            <div><b>≈ {fmtDuration(cost.seconds)}</b><span>capture</span></div>
            <div><b>≈ {fmtDuration(cost.previewSeconds)}</b><span>preview</span></div>
            <div><b>{cost.voxels >= 1e6 ? `${(cost.voxels / 1e6).toFixed(1)} M` : `${Math.round(cost.voxels / 1e3)} k`}</b><span>voxels · ~{cost.megabytes < 10 ? cost.megabytes.toFixed(1) : Math.round(cost.megabytes)} MB</span></div>
          </div>
        </Tip>
        : <div className="hint">No layered block in the scene — these settings only apply to <b>layered_block</b> objects.</div>}
    </Section>
  )
}

export function AcquisitionInspector() {
  const acq = useStudio((s) => s.exp.acquisition)
  const edit = useStudio((s) => s.edit)
  const upd = (fn: (a: typeof acq) => void, key?: string) => edit((e) => fn(e.acquisition), key ?? 'acq')
  const on = Object.values(acq.physics).filter(Boolean).length
  const total = Object.values(acq.physics).length
  return (
    <>
      <InspectorHeader icon={<SlidersHorizontal size={20} />} title="Acquisition & physics" sub={`${on}/${total} effects enabled`}
        actions={<><button className="btn sm" onClick={() => upd((a) => { for (const k of Object.keys(a.physics) as (keyof PhysicsToggles)[]) a.physics[k] = k !== 'indirect_bounce' })}>Reset</button></>} />
      <Section title="Run">
        <SF label="Mode" tip="Sample: normal capture. Dark: shutter closed (offset, dark current, read noise). White reference: uniform diffuse target for flat-fielding." value={acq.mode} options={[{ value: 'sample', label: 'Sample' }, { value: 'dark', label: 'Dark frame' }, { value: 'white_reference', label: 'White reference' }]} onChange={(v) => upd((a) => { a.mode = v })} />
        <NF label="Seed" tip="Same seed ⇒ bit-identical noise realisation." integer min={0} max={2 ** 31 - 1} step={1} value={acq.seed} bar={false} onChange={(v) => upd((a) => { a.seed = v }, 'acq:seed')} />
        <NF label="MC photons" tip="Photon packets per Monte Carlo bake. More = smoother subsurface transport, slower first run (cached afterwards)." integer min={1000} max={50_000_000} step={50000} value={acq.mc_photons} bar={false} onChange={(v) => upd((a) => { a.mc_photons = v }, 'acq:mc')} />
        <NF label="Preview scale" tip="Resolution fraction used by Preview." min={0.05} max={1} step={0.05} value={acq.preview_scale} onChange={(v) => upd((a) => { a.preview_scale = v }, 'acq:ps')} />
      </Section>
      <VolumeMcSection />
      {GROUPS.map((g) => (
        <Section key={g.title} title={g.title} defaultOpen={g.title !== 'Detector'} extra={<span className="tag">{g.items.filter(([k]) => acq.physics[k]).length}/{g.items.length}</span>}>
          <div className="toggles">
            {g.items.map(([k, label, tip]) => <Toggle key={k} disabled={k === 'indirect_bounce'} value={acq.physics[k]} label={label} title={label} tip={tip} onChange={(v) => upd((a) => { a.physics[k] = v })} />)}
          </div>
        </Section>
      ))}
      <div className="hint" style={{ padding: 12 }}>Every effect can be switched off — a teaching feature: toggle one, hit Preview and watch the Explore tab's component decomposition change.</div>
    </>
  )
}

export function ExperimentInspector() {
  const exp = useStudio((s) => s.exp)
  const edit = useStudio((s) => s.edit)
  const warnings = useStudio((s) => s.warnings)
  return (
    <>
      <InspectorHeader icon={<Layers size={20} />} title={exp.name} sub="Experiment overview — select something to edit it" />
      <Section title="Description">
        <textarea className="field" rows={3} value={exp.description} onChange={(e) => edit((x) => { x.description = e.target.value }, 'desc')} onKeyDown={(e) => e.stopPropagation()} />
        <dl className="kv" style={{ padding: 0, gridTemplateColumns: 'auto 1fr auto 1fr', gap: '3px 10px' }}>
          <dt>objects</dt><dd>{exp.objects.length}</dd><dt>materials</dt><dd>{exp.materials.length}</dd>
          <dt>sources</dt><dd>{exp.sources.length}</dd><dt>cameras</dt><dd>{exp.cameras.length}</dd>
        </dl>
      </Section>
      {warnings.length > 0 && <Section title={<>Physics notes <span className="tag">{warnings.length}</span></>}>
        {warnings.map((w, i) => <div key={i} className="hint" style={{ color: w.severity === 'warn' ? 'var(--warn)' : undefined }}>{w.severity === 'warn' ? '⚠ ' : 'ⓘ '}{w.message}</div>)}
      </Section>}
      <Section title="Shortcuts" icon={<Info size={13} />}>
        <table className="tbl"><tbody>
          {[['G / R', 'Move / rotate gizmo'], ['S', 'Toggle snap (25 mm, 15°)'], ['F', 'Frame selection'], ['C', 'Look through camera'], ['X', 'Cutaway of a layered block'], ['Space', 'Preview'], ['Del', 'Delete selected'], ['⌘Z / ⇧⌘Z', 'Undo / redo'], ['⌘D', 'Duplicate'], ['1 / 2 / 3', 'Design / Acquire / Explore'], ['[ / ]', 'Hide left / right panel']].map(([k, d]) => (
            <tr key={k}><td><span className="kbd">{k}</span></td><td className="hint">{d}</td></tr>))}
        </tbody></table>
      </Section>
      <Row label=""><Tip text="Everything on this page is stored in the .sfx.json file."><span className="hint">Autosaved to this browser · File → Save for .sfx.json</span></Tip></Row>
    </>
  )
}

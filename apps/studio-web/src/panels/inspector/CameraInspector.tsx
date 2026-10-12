import { useMemo } from 'react'
import { useStudio, activeSource } from '../../state/store'
import { Section, Row, Toggle, Num } from '../../ui/controls'
import { LineChart } from '../../ui/charts'
import { InspectorHeader, HeaderActions, NF, TF, VF, SF, BF, errAt } from './Fields'
import { CameraIcon } from '../../ui/icons'
import { gaussian, bandCentres, range } from '../../lib/spectral'
import { wavelengthToHex } from '../../lib/color'
import { cameraIntrinsics } from '../../lib/camera'
import { imagingGeometry } from '../../three/hardware/CameraFrustum'
import type { CameraSpec } from '../../generated/experiment'

/** Shape-only QE curves (the engine owns the tabulated truth). */
export function qeShape(curve: CameraSpec['sensor']['qe_curve'], wl: number): number {
  switch (curve) {
    case 'flat': return 0.8
    case 'ccd_uv': return 0.2 + 0.4 / (1 + Math.exp(-(wl - 330) / 25)) + 0.18 * Math.exp(-0.5 * ((wl - 560) / 90) ** 2) - 0.58 / (1 + Math.exp(-(wl - 960) / 55)) * 0.9
    case 'scmos': return Math.max(0, 0.82 * Math.exp(-0.5 * ((wl - 580) / 150) ** 2) + 0.05)
    default: return Math.max(0, 0.93 * Math.exp(-0.5 * ((wl - 560) / 170) ** 2) * (wl < 400 ? 0.55 + 0.45 * (wl - 300) / 100 : 1))
  }
}

export function CameraInspector({ id }: { id: string }) {
  const idx = useStudio((s) => s.exp.cameras.findIndex((x) => x.id === id))
  const cam = useStudio((s) => s.exp.cameras[idx])
  const exp = useStudio((s) => s.exp)
  const edit = useStudio((s) => s.edit)
  const active = useStudio((s) => s.exp.active_camera_id === id)
  const dupe = useStudio((s) => s.duplicateSelected), del = useStudio((s) => s.deleteSelected)
  const wlAxis = useMemo(() => range(300, 900, 2), [])
  const errors = useStudio((s) => s.errors)
  const e = (p: (string | number)[]) => errAt(errors, ['cameras', idx, ...p])
  if (!cam) return null
  const upd = (fn: (c: CameraSpec) => void, key?: string) => edit((x) => { const c = x.cameras.find((q) => q.id === id); if (c) fn(c) }, key ?? `cam:${id}`)
  const f = cam.filter, L = cam.lens, S = cam.sensor
  const intr = cameraIntrinsics(cam), g = imagingGeometry(cam)
  const ex0 = activeSource(exp).sequence.find((s) => s.enabled)?.wavelength_nm ?? null
  const centres = bandCentres(f.band_start_nm, f.band_step_nm, f.n_bands, f.band_start_policy, f.start_offset_nm, ex0)
  const lp = f.emission_longpass_nm
  const bandSeries = centres.length > 80 ? [] : centres.map((c) => ({ x: wlAxis, y: wlAxis.map((w) => gaussian(w, c, f.fwhm_nm)), color: wavelengthToHex(c), width: 1, opacity: 0.85 }))
  const series = [
    ...bandSeries,
    { name: 'QE', x: wlAxis, y: wlAxis.map((w) => qeShape(S.qe_curve, w)), color: '#e8ecf2', dash: '5 3', width: 1.6 },
    ...(lp != null ? [{ name: 'long-pass', x: wlAxis, y: wlAxis.map((w) => 1 / (1 + Math.exp(-(w - lp) / 4))), color: '#ff7b6b', width: 1.4 }] : []),
  ]
  const fieldW = g.sw * (g.s - g.f) / g.f, fieldH = g.sh * (g.s - g.f) / g.f

  return (
    <>
      <InspectorHeader icon={<CameraIcon size={20} />} title={cam.name} sub={`HSI camera · ${cam.id}${active ? ' · active' : ''}`}
        actions={<HeaderActions onDup={dupe} onDel={del} delDisabled={exp.cameras.length < 2} delTitle="Delete (needs at least one camera)" />} />
      <Section title="Camera">
        <TF label="Name" value={cam.name} onChange={(v) => upd((c) => { c.name = v }, `cam:${id}:name`)} />
        <BF label="Active" tip="Only the active source and camera are simulated." value={active} onChange={(v) => { if (v) edit((x) => { x.active_camera_id = id }) }} />
        <div className="kv" style={{ padding: '2px 0 0' }}>
          <dt>vertical FOV</dt><dd>{intr.fovYDeg.toFixed(2)}°</dd>
          <dt>field at focus</dt><dd>{fieldW.toFixed(1)} × {fieldH.toFixed(1)} mm</dd>
          <dt>pixel on object</dt><dd>{(fieldW / S.width_px * 1000).toFixed(0)} µm</dd>
          <dt>DOF</dt><dd>{g.near.toFixed(0)} – {Number.isFinite(g.far) ? g.far.toFixed(0) : '∞'} mm</dd>
        </div>
      </Section>
      <Section title="Pose">
        <VF label="Position" value={cam.position_mm} step={1} unit="mm" onChange={(v) => upd((c) => { c.position_mm = v }, `cam:${id}:pos`)} />
        <VF label="Target" value={cam.target_mm} step={1} unit="mm" onChange={(v) => upd((c) => { c.target_mm = v }, `cam:${id}:tgt`)} />
        <NF label="Roll" value={cam.roll_deg} min={-180} max={180} step={1} unit="°" onChange={(v) => upd((c) => { c.roll_deg = v }, `cam:${id}:roll`)} />
      </Section>
      <Section title="Lens">
        <NF label="Focal length" value={L.focal_length_mm} min={4} max={400} step={0.5} unit="mm" err={e(['lens', 'focal_length_mm'])} onChange={(v) => upd((c) => { c.lens.focal_length_mm = v }, `cam:${id}:fl`)} />
        <NF label="f-number" tip="Aperture: sets depth of field and diffraction blur (Airy σ ≈ 0.42·λ·N)." value={L.f_number} min={0.6} max={32} step={0.1} err={e(['lens', 'f_number'])} onChange={(v) => upd((c) => { c.lens.f_number = v }, `cam:${id}:fn`)} />
        <NF label="Focus distance" value={L.focus_distance_mm} min={20} max={5000} step={5} unit="mm" onChange={(v) => upd((c) => { c.lens.focus_distance_mm = v }, `cam:${id}:fd`)} />
        <NF label="Transmission" value={L.transmission} min={0.05} max={1} step={0.01} onChange={(v) => upd((c) => { c.lens.transmission = v }, `cam:${id}:tr`)} />
        <NF label="Axial chromatic" tip="Focal shift per 100 nm away from 550 nm." value={L.axial_chromatic_mm_per_100nm} min={0} max={2} step={0.01} unit="mm" onChange={(v) => upd((c) => { c.lens.axial_chromatic_mm_per_100nm = v }, `cam:${id}:ca`)} />
      </Section>
      <Section title="Sensor" defaultOpen={false}>
        <div className="grid2"><NF label="Width" integer min={8} max={4096} step={1} value={S.width_px} unit="px" onChange={(v) => upd((c) => { c.sensor.width_px = v }, `cam:${id}:w`)} />
          <NF label="Height" integer min={8} max={4096} step={1} value={S.height_px} unit="px" onChange={(v) => upd((c) => { c.sensor.height_px = v }, `cam:${id}:h`)} /></div>
        <NF label="Pixel pitch" value={S.pixel_pitch_um} min={0.5} max={40} step={0.05} unit="µm" onChange={(v) => upd((c) => { c.sensor.pixel_pitch_um = v }, `cam:${id}:pp`)} />
        <NF label="Binning" integer min={1} max={16} step={1} value={S.binning} tip="Effective pixel pitch = pitch × binning. width/height are output (binned) pixels." onChange={(v) => upd((c) => { c.sensor.binning = v }, `cam:${id}:bin`)} />
        <SF label="QE curve" value={S.qe_curve} options={['ccd_bv', 'ccd_uv', 'scmos', 'flat'] as const} onChange={(v) => upd((c) => { c.sensor.qe_curve = v })} />
        <SF label="Bit depth" value={S.bit_depth} options={[12, 14, 16] as const} onChange={(v) => upd((c) => { c.sensor.bit_depth = v })} />
        <NF label="Full well" value={S.full_well_e} min={100} max={1e6} step={500} unit="e⁻" onChange={(v) => upd((c) => { c.sensor.full_well_e = v }, `cam:${id}:fw`)} />
        <NF label="Read noise" value={S.read_noise_e} min={0} max={100} step={0.1} unit="e⁻" onChange={(v) => upd((c) => { c.sensor.read_noise_e = v }, `cam:${id}:rn`)} />
        <NF label="Dark current" value={S.dark_current_e_per_s} min={0} max={1000} step={0.1} unit="e⁻/s" onChange={(v) => upd((c) => { c.sensor.dark_current_e_per_s = v }, `cam:${id}:dc`)} />
        <NF label="Gain" value={S.gain_e_per_adu} min={0.05} max={50} step={0.05} unit="e⁻/ADU" onChange={(v) => upd((c) => { c.sensor.gain_e_per_adu = v }, `cam:${id}:gain`)} />
        <NF label="Offset" value={S.offset_adu} min={0} max={5000} step={1} unit="ADU" onChange={(v) => upd((c) => { c.sensor.offset_adu = v }, `cam:${id}:off`)} />
        <NF label="PRNU" tip="Photo-response non-uniformity, 1σ." value={S.prnu_pct} min={0} max={10} step={0.1} unit="%" onChange={(v) => upd((c) => { c.sensor.prnu_pct = v }, `cam:${id}:prnu`)} />
        <NF label="DSNU" tip="Dark-signal non-uniformity, 1σ." value={S.dsnu_e} min={0} max={50} step={0.1} unit="e⁻" onChange={(v) => upd((c) => { c.sensor.dsnu_e = v }, `cam:${id}:dsnu`)} />
        <NF label="Hot pixels" value={S.hot_pixel_fraction} min={0} max={0.1} step={0.00005} precision={5} onChange={(v) => upd((c) => { c.sensor.hot_pixel_fraction = v }, `cam:${id}:hot`)} />
      </Section>
      <Section title="Spectral filter">
        <SF label="Type" tip="LCTF: liquid-crystal tunable filter (Nuance-like). Filter wheel: interference band-pass (angle-dependent blue shift)." value={f.kind} options={[{ value: 'lctf', label: 'LCTF' }, { value: 'filter_wheel', label: 'Filter wheel' }]} onChange={(v) => upd((c) => { c.filter.kind = v })} />
        <NF label="Band start" value={f.band_start_nm} min={200} max={1100} step={1} unit="nm" onChange={(v) => upd((c) => { c.filter.band_start_nm = v }, `cam:${id}:bs`)} />
        <NF label="Band step" value={f.band_step_nm} min={0.5} max={100} step={0.5} unit="nm" onChange={(v) => upd((c) => { c.filter.band_step_nm = v }, `cam:${id}:bst`)} />
        <NF label="Bands" integer min={1} max={512} step={1} value={f.n_bands} onChange={(v) => upd((c) => { c.filter.n_bands = v }, `cam:${id}:nb`)} />
        <SF label="Start policy" tip="'excitation_offset' starts the band list at max(start, excitation + offset): the real Nuance protocol for 415/430 nm excitation." value={f.band_start_policy} options={[{ value: 'fixed', label: 'fixed' }, { value: 'excitation_offset', label: 'excitation + offset' }]} onChange={(v) => upd((c) => { c.filter.band_start_policy = v })} />
        {f.band_start_policy === 'excitation_offset' && <NF label="Start offset" value={f.start_offset_nm} min={0} max={200} step={1} unit="nm" onChange={(v) => upd((c) => { c.filter.start_offset_nm = v }, `cam:${id}:so`)} />}
        <NF label="Passband FWHM" value={f.fwhm_nm} min={1} max={100} step={0.5} unit="nm" onChange={(v) => upd((c) => { c.filter.fwhm_nm = v }, `cam:${id}:pf`)} />
        <SF label="Peak T curve" value={f.peak_transmission_curve} options={['lctf_vis', 'flat'] as const} onChange={(v) => upd((c) => { c.filter.peak_transmission_curve = v })} />
        {f.kind === 'filter_wheel' && <NF label="n_eff" tip="Effective index in the angle shift λ(θ) = λ₀·√(1 − sin²θ/n_eff²)." value={f.interference_n_eff} min={1.01} max={4} step={0.05} onChange={(v) => upd((c) => { c.filter.interference_n_eff = v })} />}
        <Row label="Long-pass" tip="Emission long-pass edge that blocks the excitation line."><div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <Toggle value={lp != null} onChange={(v) => upd((c) => { c.filter.emission_longpass_nm = v ? (ex0 ?? 400) + 20 : null })} />
          {lp != null && <Num value={lp} min={200} max={1100} step={1} unit="nm" bar={false} onChange={(v) => upd((c) => { c.filter.emission_longpass_nm = v }, `cam:${id}:lp`)} />}</div></Row>
        {lp != null && <NF label="Long-pass OD" value={f.longpass_od} min={0} max={8} step={0.1} onChange={(v) => upd((c) => { c.filter.longpass_od = v })} />}
      </Section>
      <Section title="Passbands · QE · long-pass">
        <div className="hint" style={{ marginTop: -4 }}>{ex0 != null ? `Bands for the first excitation (${ex0} nm): ${centres[0]?.toFixed(0)}–${centres.at(-1)?.toFixed(0)} nm, ${centres.length} bands.` : 'No enabled excitation.'}</div>
        <LineChart series={series} height={130} xDomain={[300, 900]} yDomain={[0, 1.05]} strip xLabel="nm" legend
          markers={ex0 != null ? [{ x: ex0, color: wavelengthToHex(ex0), label: 'ex' }] : []} />
      </Section>
    </>
  )
}

import { useRef } from 'react'
import { AlertTriangle } from 'lucide-react'
import { Section, Toggle } from '../../ui/controls'
import { NF, SF } from './Fields'
import { DEFAULT_TISSUE, tissueWarnings } from '../../lib/tissueSpectra'
import type { MaterialSpec, TissueComposition } from '../../generated/experiment'

type Upd = (fn: (m: MaterialSpec) => void, key?: string) => void

/** Scattering model selector + parameters (none / power law / Mie + Rayleigh, Jacques 2013 eq. 2). */
export function ScatteringEditor({ mat, id, upd }: { mat: MaterialSpec; id: string; upd: Upd }) {
  const s = mat.scattering
  return (
    <>
      <SF label="Scattering" tip="none: clear medium. Power law: μs′ = a·(λ/500)^−b. Mie + Rayleigh (Jacques 2013 eq. 2): μs′ = a·[f·(λ/500)^−4 + (1−f)·(λ/500)^−b_Mie], with Rayleigh weight f from sub-wavelength fibrils."
        value={s.model} options={[{ value: 'none', label: 'None (clear)' }, { value: 'power_law', label: 'Power law' }, { value: 'mie_rayleigh', label: 'Mie + Rayleigh' }]}
        onChange={(v) => upd((m) => { m.scattering.model = v; if (v !== 'none' && m.scattering.mus_prime_500_per_mm === 0) m.scattering.mus_prime_500_per_mm = 1 })} />
      {s.model !== 'none' && <>
        <NF label="μs′ (500 nm)" tip="Reduced scattering coefficient at 500 nm, a′ in Jacques 2013." value={s.mus_prime_500_per_mm} min={0} max={50} step={0.05} unit="1/mm" onChange={(v) => upd((m) => { m.scattering.mus_prime_500_per_mm = v }, `mat:${id}:mus`)} />
        <NF label={s.model === 'mie_rayleigh' ? 'Mie power b' : 'Power b'} tip="Scattering power (wavelength exponent) of the Mie / power-law component. Typical tissue 0.5–1.5." value={s.power_b} min={0} max={4} step={0.05} onChange={(v) => upd((m) => { m.scattering.power_b = v }, `mat:${id}:b`)} />
        {s.model === 'mie_rayleigh' && <NF label="Rayleigh f" tip="Fraction of μs′(500 nm) that scatters as λ⁻⁴ (small collagen fibrils, organelles). 0 = pure Mie power law, 1 = pure Rayleigh." value={s.f_rayleigh} min={0} max={1} step={0.01} onChange={(v) => upd((m) => { m.scattering.f_rayleigh = v }, `mat:${id}:fray`)} />}
        <NF label="Anisotropy g" tip="Henyey–Greenstein anisotropy; μs = μs′/(1−g)." value={s.g} min={-0.99} max={0.99} step={0.01} onChange={(v) => upd((m) => { m.scattering.g = v }, `mat:${id}:g`)} />
      </>}
    </>
  )
}

const pct = (v: number) => +(v * 100).toFixed(4)

/** Chromophore composition (Jacques 2013 eq. 1) with an on/off switch (`tissue` null ↔ object). */
export function TissueEditor({ mat, id, upd }: { mat: MaterialSpec; id: string; upd: Upd }) {
  const t = mat.tissue
  const last = useRef<TissueComposition | null>(null)
  if (t) last.current = t
  const set = (fn: (t: TissueComposition) => void, key: string) => upd((m) => { if (m.tissue) fn(m.tissue) }, `mat:${id}:${key}`)
  const warns = t ? tissueWarnings(t) : []
  return (
    <Section title="Tissue composition" defaultOpen={!!t}
      extra={<Toggle value={!!t} title="Compose μa from tissue chromophores" onChange={(on) => upd((m) => { m.tissue = on ? structuredClone(last.current ?? DEFAULT_TISSUE) : null })} />}>
      {!t && <div className="hint">Off. Switch on to build the absorption from blood (HbO₂ / Hb), water, lipid, melanosomes, bilirubin and β-carotene. It is <i>added</i> to the dyes and the background absorber.</div>}
      {t && <>
        <div className="grp-title">Blood</div>
        <NF label="Blood volume" tip="Blood volume fraction B: share of the tissue volume filled with whole blood." value={pct(t.blood_volume_fraction)} min={0} max={100} step={0.1} unit="%" onChange={(v) => set((x) => { x.blood_volume_fraction = v / 100 }, 'B')} />
        <NF label="StO₂" tip="Oxygen saturation S = HbO₂/(HbO₂+Hb). Arterial ≈ 97 %, venous ≈ 65 %, tumours often < 60 %." value={pct(t.oxygen_saturation)} min={0} max={100} step={0.5} unit="%" onChange={(v) => set((x) => { x.oxygen_saturation = v / 100 }, 'S')} />
        <NF label="[Hb] in blood" tip="Haemoglobin concentration of whole blood (g/L); 150 g/L is typical for adults (64 500 g/mol per tetramer)." value={t.hemoglobin_g_per_L} min={20} max={260} step={1} unit="g/L" onChange={(v) => set((x) => { x.hemoglobin_g_per_L = v }, 'hb')} />
        <div className="grp-title">Matrix</div>
        <NF label="Water" tip="Water volume fraction W (Hale & Querry). Soft tissue 60–80 %." value={pct(t.water_fraction)} min={0} max={100} step={0.5} unit="%" onChange={(v) => set((x) => { x.water_fraction = v / 100 }, 'W')} />
        <NF label="Lipid" tip="Lipid (fat) volume fraction F (van Veen 2005). Subcutis ≈ 70–90 %." value={pct(t.lipid_fraction)} min={0} max={100} step={0.5} unit="%" onChange={(v) => set((x) => { x.lipid_fraction = v / 100 }, 'F')} />
        <NF label="Melanosomes" tip="Melanosome volume fraction M (Jacques): epidermis 1–43 % depending on skin type; μa = M·6.6·10¹¹·λ⁻³·³³ cm⁻¹." value={pct(t.melanosome_fraction)} min={0} max={100} step={0.1} unit="%" onChange={(v) => set((x) => { x.melanosome_fraction = v / 100 }, 'M')} />
        <div className="grp-title">Pigments</div>
        <NF label="Bilirubin" tip="Bilirubin concentration (µM); jaundice ≈ 100–400 µM. Absorbs at ≈ 450 nm." value={t.bilirubin_uM} min={0} max={1000} step={1} unit="µM" onChange={(v) => set((x) => { x.bilirubin_uM = v }, 'bili')} />
        <NF label="β-carotene" tip="β-carotene concentration (µM); skin carotenoids ≈ 0–10 µM. Absorbs 420–500 nm." value={t.beta_carotene_uM} min={0} max={200} step={0.1} unit="µM" onChange={(v) => set((x) => { x.beta_carotene_uM = v }, 'bcar')} />
        {warns.map((w, i) => <div key={i} className="lyr-issue" style={{ color: 'var(--warn)', background: 'rgba(255,190,70,.08)', borderColor: 'rgba(255,190,70,.35)' }}><AlertTriangle size={13} />{w}</div>)}
      </>}
    </Section>
  )
}

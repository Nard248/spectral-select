import { useMemo, useState } from 'react'
import { Section, Num, Tip } from '../../ui/controls'
import { LineChart, type Series } from '../../ui/charts'
import { wavelengthToHex } from '../../lib/color'
import { materialMua, musPrime, range } from '../../lib/spectral'
import { penetrationDepth, tissueMuaTerms, WL_MAX, WL_MIN } from '../../lib/tissueSpectra'
import type { FluorophoreSpec, MaterialSpec } from '../../generated/experiment'
import { interp, useMaterialOptics } from '../../research/useMaterialOptics'
import { useStudio } from '../../state/store'

const COMPONENTS: { key: 'hbo2' | 'hb' | 'water' | 'lipid' | 'melanosome' | 'bilirubin' | 'carotene'; name: string; color: string }[] = [
  { key: 'hbo2', name: 'HbO₂', color: '#ff6b6b' }, { key: 'hb', name: 'Hb', color: '#7aa2ff' }, { key: 'water', name: 'water', color: '#4fd8e8' },
  { key: 'lipid', name: 'lipid', color: '#f4d35e' }, { key: 'melanosome', name: 'melanosome', color: '#c08a5a' }, { key: 'bilirubin', name: 'bilirubin', color: '#ffb347' },
  { key: 'carotene', name: 'β-carotene', color: '#ff8c42' },
]
/** engine component key → label and colour (dyes `dye:<name>` and the background absorber are added on the fly) */
const ENGINE_COMPONENTS: Record<string, { name: string; color: string }> = {
  hbo2: { name: 'HbO₂', color: '#ff6b6b' }, hb: { name: 'Hb', color: '#7aa2ff' }, water: { name: 'water', color: '#4fd8e8' }, lipid: { name: 'lipid', color: '#f4d35e' },
  melanosome: { name: 'melanosome', color: '#c08a5a' }, bilirubin: { name: 'bilirubin', color: '#ffb347' }, beta_carotene: { name: 'β-carotene', color: '#ff8c42' }, absorber: { name: 'absorber', color: '#b0a4ff' },
}
const fmt = (v: number) => (v === 0 ? '0' : v >= 100 ? v.toFixed(0) : v >= 1 ? v.toFixed(2) : v >= 0.01 ? v.toFixed(3) : v.toExponential(1))
const fmtDepth = (d: number) => (!Number.isFinite(d) ? '∞' : d >= 10 ? `${d.toFixed(1)} mm` : d >= 0.1 ? `${d.toFixed(2)} mm` : `${(d * 1000).toFixed(0)} µm`)

/** Live μa(λ) (log) and μs′(λ) of the material over 350–900 nm, plus penetration depth δ = 1/√(3 μa (μa+μs′)) at the sequence
 *  excitations and a free probe wavelength. Everything is computed client-side from `lib/tissueSpectra`. */
export function OpticalProperties({ mat, fl, excitations }: { mat: MaterialSpec; fl: Record<string, FluorophoreSpec>; excitations: number[] }) {
  const [probe, setProbe] = useState(650)
  const customDyes = useStudio((s) => s.exp.fluorophores)
  const eng = useMaterialOptics(mat, customDyes, WL_MIN, WL_MAX, 5)
  const live = eng.source === 'engine' ? eng.data : null
  const wl = useMemo(() => range(WL_MIN, WL_MAX, 5), [])
  const muaTotal = useMemo(() => (live ? live.mua_per_mm : wl.map((w) => materialMua(mat, fl, w))), [mat, fl, wl, live])
  const musp = useMemo(() => (live ? live.mus_prime_per_mm : wl.map((w) => musPrime(mat.scattering, w))), [mat.scattering, wl, live])
  const terms = useMemo(() => (mat.tissue ? wl.map((w) => tissueMuaTerms(mat.tissue!, w)) : null), [mat.tissue, wl])
  const xs = live ? live.nm : wl
  const floor = 1.2e-4
  const series: Series[] = [
    ...(live
      ? Object.entries(live.components).filter(([k]) => ENGINE_COMPONENTS[k] || k.startsWith('dye:')).filter(([, v]) => v.some((y) => y > 0)).map(([k, v]) => {
        const dye = k.startsWith('dye:') ? k.slice(4) : null
        const meta = dye ? { name: dye, color: fl[dye]?.display_color ?? wavelengthToHex(fl[dye]?.em_peak_nm ?? 550) } : ENGINE_COMPONENTS[k]
        return { name: meta.name, x: xs, y: v.map((y) => Math.max(y, floor)), color: meta.color, width: 1.1, opacity: 0.85, dash: '3 2' }
      })
      : terms ? COMPONENTS.filter((c) => terms.some((t) => t[c.key] > 0)).map((c) => ({
        name: c.name, x: wl, y: terms.map((t) => Math.max(t[c.key], floor)), color: c.color, width: 1.1, opacity: 0.85, dash: '3 2',
      })) : []),
    { name: 'total μa', x: xs, y: muaTotal.map((v) => Math.max(v, floor)), color: '#e8ecf2', fill: true, width: 2 },
  ]
  const scat = mat.scattering.model !== 'none'
  const exs = [...new Set(excitations)].slice(0, 4)
  const rows = [...exs.map((e) => ({ nm: e, probe: false })), { nm: probe, probe: true }].map((r) => {
    const a = live ? interp(live.nm, live.mua_per_mm, r.nm) : materialMua(mat, fl, r.nm), s = live ? interp(live.nm, live.mus_prime_per_mm, r.nm) : musPrime(mat.scattering, r.nm)
    return { ...r, mua: a, musp: s, delta: penetrationDepth(a, s) }
  })
  return (
    <Section title="Optical properties" extra={<Tip title={live ? 'Engine tables' : 'Client approximation'} text={live ? 'Computed by the Python engine from the same absorption and scattering tables the simulator uses (Prahl haemoglobin, Hale & Querry water, van Veen lipid, Jacques 2013).' : eng.error ? `The engine did not answer (${eng.error}); showing the built-in approximate tables (within ~10-15 % of the engine).` : 'Built-in approximate tables (within ~10-15 % of the engine). Start the engine for the exact spectra.'}><span className="tag" style={live ? { color: 'var(--ok)', background: 'rgba(78,242,138,.1)' } : undefined}>350–900 nm · {live ? 'engine' : eng.loading ? 'updating…' : 'approx.'}</span></Tip>}>
      <div className="hint" style={{ marginTop: -2 }}>Absorption μa(λ) (1/mm, log): dyes + background absorber{mat.tissue ? ' + tissue chromophores' : ''}. Coloured ticks mark the sequence excitations.</div>
      <LineChart series={series} height={196} xDomain={[WL_MIN, WL_MAX]} yDomain={[-4, 2]} logY strip xLabel="nm" yLabel="μa 1/mm"
        markers={exs.map((e) => ({ x: e, color: wavelengthToHex(e) }))} legend />
      <div className="hint" style={{ marginTop: 2 }}>Reduced scattering μs′(λ) (1/mm){mat.scattering.model === 'mie_rayleigh' ? ' · Mie + Rayleigh' : mat.scattering.model === 'power_law' ? ' · power law' : ''}</div>
      {scat
        ? <LineChart series={[{ x: xs, y: musp, color: '#9bb4ff', fill: true }]} height={84} xDomain={[WL_MIN, WL_MAX]} yLabel="μs′ 1/mm" markers={exs.map((e) => ({ x: e, color: wavelengthToHex(e) }))} />
        : <div className="hint" style={{ padding: '6px 0' }}>Scattering model is “none”: μs′ = 0 (clear medium).</div>}
      <Tip text="Diffusion-theory 1/e penetration depth δ = 1/√(3·μa·(μa+μs′)). It assumes μs′ ≫ μa; it is a guide for how deep the light field reaches, the Monte Carlo is the authority.">
        <div className="hint" style={{ marginTop: 4 }}>Penetration depth δ = 1/√(3 μa (μa+μs′))</div>
      </Tip>
      <table className="opt-table">
        <thead><tr><th>λ</th><th>μa</th><th>μs′</th><th>δ</th></tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={r.probe ? 'probe' : ''}>
              <td>{r.probe
                ? <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}><i style={{ background: wavelengthToHex(r.nm) }} /><span style={{ width: 82 }}><Num value={probe} min={WL_MIN} max={WL_MAX} step={5} integer bar={false} unit="nm" onChange={setProbe} /></span></span>
                : <><i style={{ background: wavelengthToHex(r.nm) }} />{r.nm} nm</>}</td>
              <td>{fmt(r.mua)}</td><td>{fmt(r.musp)}</td><td><b>{fmtDepth(r.delta)}</b></td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="hint">μa, μs′ in 1/mm. The last row is a free probe wavelength.</div>
    </Section>
  )
}

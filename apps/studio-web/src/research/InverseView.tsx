import { useEffect, useMemo, useState } from 'react'
import { Atom, Crosshair, Gauge, Grid3x3, Play, Plus, Square, Target, X, Sigma } from 'lucide-react'
import { useStudio } from '../state/store'
import { enumerateParams } from '../sweep/paths'
import { ParamPicker } from '../sweep/SweepPanel'
import { Num, Sel, Seg, Tip, Toggle } from '../ui/controls'
import { XYChart, fmtVal } from '../ui/plots'
import { wavelengthToHex } from '../lib/color'
import type { EndmemberSource, UnmixMethod } from '../api/researchTypes'
import { RoiPick } from './StudyBuilder'
import { C_NOISE, C_OK, C_S1, C_ST, ChannelMap, MapCanvas } from './charts'
import { channelGrid, designBaselines } from './logic'
import { useResearch, type Busy } from './store'
import { expChannels } from './actions'
import {
  addInvParam, buildSurrogate, cancelInverseJob, currentJob, defaultCapture, patchInvParam, removeInvParam, runDesign, runFisher, runFit, runFitMap, runUnmix,
} from './inverseActions'

// ---------------------------------------------------------------------------------------------- shared setup column
function Setup() {
  const exp = useStudio((s) => s.exp)
  const captures = useStudio((s) => s.captures)
  const inv = useResearch((s) => s.inv)
  const setInv = useResearch((s) => s.setInv)
  const [open, setOpen] = useState(false)
  const all = useMemo(() => enumerateParams(exp), [exp])
  const paths = useMemo(() => all.map((o) => o.path), [all])
  const ch = expChannels(exp)
  const capId = inv.captureId ?? defaultCapture()
  const lo = ch.exs.length ? ch.bands(ch.exs[0])[0] : 400
  const hi = ch.exs.length ? ch.bands(ch.exs[0]).slice(-1)[0] : 700
  useEffect(() => {
    if (inv.params.length) return
    const th = all.find((p) => p.group === 'Layer thickness' && /collagen|cover/i.test(p.layer ?? '')) ?? all.find((p) => p.group === 'Layer thickness')
    if (th) addInvParam(th, paths)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [all])
  return (
    <div className="cfg rs-cfg">
      <div className="rs-title"><Sigma size={14} /><span>Inverse problem</span><small>what can the instrument tell us?</small></div>
      <div className="sec-h">Parameters to recover</div>
      <div className="rs-sec">
        {inv.params.map((p, i) => (
          <div key={p.path} className="rs-card">
            <div className="rs-card-h"><input className="rs-name" value={p.label} readOnly aria-label="Parameter" /><button className="btn ghost icon sm" aria-label={`Remove ${p.label}`} onClick={() => removeInvParam(i)}><X size={13} /></button></div>
            <div className="rs-lohi">
              <div><div className="hint">low</div><Num value={p.lo} step={Math.abs(p.hi - p.lo) / 50 || 0.01} bar={false} unit={p.unit} onChange={(v) => patchInvParam(i, { lo: v })} /></div>
              <div><div className="hint">high</div><Num value={p.hi} step={Math.abs(p.hi - p.lo) / 50 || 0.01} bar={false} unit={p.unit} onChange={(v) => patchInvParam(i, { hi: v })} /></div>
              <Tip text="Absolute: the parameter takes the value itself. Shift: the swept value is added to the reference (keeps e.g. a wedge's profile)."><span style={{ alignSelf: 'end' }}><Seg small value={p.mode} onChange={(mode) => patchInvParam(i, { mode })} options={[{ value: 'absolute', label: 'abs' }, { value: 'shift', label: 'shift' }]} /></span></Tip>
            </div>
            {p.paths.length > 1 && <div className="hint">linked: {p.paths.length - 1} extra pointer{p.paths.length > 2 ? 's' : ''} move together</div>}
          </div>))}
        <div style={{ position: 'relative' }}>
          <button className="btn sm" style={{ width: '100%' }} onClick={() => setOpen(!open)} aria-expanded={open}><Plus size={13} />Add parameter…</button>
          {open && <ParamPicker params={all.filter((o) => !inv.params.some((p) => p.path === o.path))} value={null} onPick={(o) => { addInvParam(o, paths); setOpen(false) }} onClose={() => setOpen(false)} />}
        </div>
        <div className="hint">Up to 8. Thickness, concentration, scattering … anything the forward model exposes.</div>
      </div>
      <div className="sec-h">Observation</div>
      <div className="rs-sec">
        <div><div className="hint">capture (for unmixing and fits)</div>
          <select className="field" aria-label="Capture" value={capId ?? ''} onChange={(e) => setInv({ captureId: e.target.value || null })}>
            {captures.length === 0 && <option value="">— no capture yet —</option>}
            {captures.map((c) => <option key={c.id} value={c.id}>{c.name} · {c.id} · {c.width}×{c.height}</option>)}
          </select></div>
        <RoiPick label="ROI (fractions of the image)" value={inv.roi} onChange={(roi) => setInv({ roi })} />
        <div>
          <div className="hint" style={{ paddingBottom: 3 }}>Excitations</div>
          <div className="rs-exs">{ch.exs.map((e) => { const on = !inv.excitations || inv.excitations.includes(e); return <button key={e} aria-pressed={on} className={`rs-ex${on ? ' on' : ''}`} style={{ ['--c' as string]: wavelengthToHex(e) }}
            onClick={() => { const cur = inv.excitations ?? ch.exs; const nx = on ? cur.filter((x) => x !== e) : [...cur, e].sort((a, b) => a - b); setInv({ excitations: nx.length === 0 || nx.length === ch.exs.length ? null : nx }) }}>{e}</button> })}</div>
        </div>
        <div className="rs-two">
          <div><div className="hint">emission from</div><Num value={inv.bandFrom ?? lo} step={5} integer bar={false} unit="nm" onChange={(v) => setInv({ bandFrom: v })} /></div>
          <div><div className="hint">to</div><Num value={inv.bandTo ?? hi} step={5} integer bar={false} unit="nm" onChange={(v) => setInv({ bandTo: v })} /></div>
        </div>
        {(inv.bandFrom != null || inv.bandTo != null) && <button className="btn sm ghost" onClick={() => setInv({ bandFrom: null, bandTo: null })}>All emission bands</button>}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------------------------- small shared bits
function RunStatus({ b, label, onCancel }: { b: Busy<unknown>; label: string; onCancel?: () => void }) {
  if (b.state === 'idle') return null
  return (
    <div className="rs-status" role="status">
      {b.state === 'running' && <><span className="spinner" style={{ width: 12, height: 12, margin: 0, borderWidth: 1.5 }} /><span>{b.message || label}</span><div className="progress"><i style={{ width: `${Math.round(b.fraction * 100)}%` }} /></div>{onCancel && <button className="btn sm ghost" onClick={onCancel}><Square size={11} />Cancel</button>}</>}
      {b.state === 'error' && <div className="note err" style={{ margin: 0, flex: 1 }}>{b.error}</div>}
    </div>
  )
}
const fmtSd = (v: number | null | undefined) => (v == null || !Number.isFinite(v) ? '∞' : fmtVal(v))

// ---------------------------------------------------------------------------------------------- unmixing
function UnmixPanel() {
  const inv = useResearch((s) => s.inv)
  const setInv = useResearch((s) => s.setInv)
  const r = inv.unmix.result
  const names = r?.names ?? []
  const COLORS = ['#35d8ff', '#9b8cff', '#4ef28a', '#ffb547', '#ff7a8a', '#7cc4ff']
  const em = r?.endmembers.channels
  const exs = em ? [...new Set(em.ex_nm)] : []
  const [exSel, setExSel] = useState<number | null>(null)
  const exv = exSel ?? exs[0]
  return (
    <div className="lf-card">
      <header><Atom size={14} style={{ color: 'var(--accent)' }} />Spectral unmixing <small>abundance maps from the captured cube</small><span className="sp" />
        <button className="btn sm primary" disabled={inv.unmix.state === 'running'} onClick={() => void runUnmix()}><Play size={12} />Unmix</button></header>
      <div className="cbody">
        <div className="rs-opts">
          <div><div className="hint">method</div><Sel value={inv.unmixMethod} onChange={(v: UnmixMethod) => setInv({ unmixMethod: v })} options={[{ value: 'nnls', label: 'NNLS (non-negative)' }, { value: 'fcls', label: 'FCLS (sum-to-one)' }, { value: 'ls', label: 'Least squares' }]} /></div>
          <div><div className="hint">endmember source</div><Sel value={inv.endmemberSource} onChange={(v: EndmemberSource) => setInv({ endmemberSource: v })} options={[{ value: 'forward_model', label: 'Forward model (known dyes)' }, { value: 'rois', label: 'ROIs (Explorer)' }, { value: 'nmf', label: 'NMF (blind)' }, { value: 'vca', label: 'VCA (blind)' }]} /></div>
          {(inv.endmemberSource === 'nmf' || inv.endmemberSource === 'vca') && <div><div className="hint">components</div><Num value={inv.nComponents} min={2} max={8} integer bar={false} onChange={(v) => setInv({ nComponents: v })} /></div>}
          <div><div className="hint">noise weighting</div><Seg small value={inv.weighting} onChange={(v) => setInv({ weighting: v })} options={[{ value: 'none', label: 'None' }, { value: 'poisson', label: 'Poisson' }]} /></div>
          {inv.endmemberSource === 'forward_model' && <Toggle value={inv.compareGt} onChange={(v) => setInv({ compareGt: v })} label="Compare with ground truth" tip="Regress the abundances of 60 random pixels on the simulator's true component amounts." />}
        </div>
        <RunStatus b={inv.unmix} label="unmixing…" />
        {!r && inv.unmix.state === 'idle' && <div className="hint" style={{ padding: '14px 0 4px' }}>Decompose every pixel into endmember spectra. With the forward model the endmembers are the dyes of the scene, so the abundances can be checked against the simulator’s ground truth.</div>}
        {r && (
          <>
            <div className="rs-facts" style={{ marginTop: 8 }}>
              <span>method <b>{r.method}</b></span><span>residual <b>{r.mean_residual_fraction != null ? `${(r.mean_residual_fraction * 100).toFixed(1)} %` : '—'}</b></span><span>{r.shape[1]}×{r.shape[0]} px</span><span>{names.length} endmembers</span>
            </div>
            <div className="rs-maps">
              {names.map((n, k) => { const m = inv.unmixMaps[n]; const gt = r.gt_comparison?.[n]; return (
                <figure key={n} className="rs-map">
                  <figcaption><i style={{ background: COLORS[k % COLORS.length] }} />{n}<span>{fmtVal(r.mean_abundance[n] ?? NaN)}</span></figcaption>
                  {m ? <MapCanvas data={m.data} w={m.w} h={m.h} cmap="viridis" maxHeight={150} label={`abundance of ${n}`} /> : <div className="hint">no map</div>}
                  {gt && <div className="rs-gt"><span title="Pearson correlation with the true amount">r <b>{gt.pearson_r?.toFixed(3) ?? '—'}</b></span><span title="Slope of estimated on true">slope <b>{gt.slope?.toFixed(2) ?? '—'}</b></span><span title="Relative RMSE">rRMSE <b>{gt.rel_rmse != null ? `${(gt.rel_rmse * 100).toFixed(1)} %` : '—'}</b></span></div>}
                </figure>) })}
              {inv.unmixMaps.rmse && <figure className="rs-map"><figcaption><i style={{ background: '#ff7a8a' }} />RMSE map<span>reconstruction error</span></figcaption><MapCanvas data={inv.unmixMaps.rmse.data} w={inv.unmixMaps.rmse.w} h={inv.unmixMaps.rmse.h} cmap="magma" maxHeight={150} label="reconstruction RMSE" unit="ADU" /></figure>}
            </div>
            {em && em.em_nm.length > 0 && (
              <>
                <div className="hint" style={{ paddingTop: 10 }}>Endmember spectra (unit-sum){exs.length > 1 ? ' at excitation' : ''}</div>
                {exs.length > 1 && <div className="chips2" style={{ padding: '4px 0' }}>{exs.map((e) => <button key={e} className={`chip2 mono${e === exv ? ' on' : ''}`} style={{ ['--c' as string]: wavelengthToHex(e) }} onClick={() => setExSel(e)}><i />{e} nm</button>)}</div>}
                <XYChart height={170} zeroBase xLabel="emission (nm)" yLabel="relative" legend
                  series={names.map((n, k) => { const idx = em.ex_nm.map((e, c) => (e === exv ? c : -1)).filter((c) => c >= 0); return { name: n, x: idx.map((c) => em.em_nm[c]), y: idx.map((c) => r.endmembers.matrix[k][c]), color: COLORS[k % COLORS.length], width: 2 } })} />
              </>)}
          </>)}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------------------------- Fisher information
function useChannelValue(which: number) {
  const fr = useResearch((s) => s.inv.fisher.result)
  return useMemo(() => {
    if (!fr) return null
    const C = fr.n_channels, P = fr.names.length
    const v = new Array<number>(C).fill(0)
    if (which < 0) {
      const tot = new Array<number>(P).fill(0)
      fr.info_per_channel.forEach((row) => row.forEach((x, p) => { tot[p] += x }))
      fr.info_per_channel.forEach((row, c) => { v[c] = row.reduce((a, x, p) => a + (tot[p] > 0 ? x / tot[p] : 0), 0) / P })
    } else fr.info_per_channel.forEach((row, c) => { v[c] = row[which] })
    return { v, grid: channelGrid(fr.channels, v) }
  }, [fr, which])
}

function InfoPanel() {
  const inv = useResearch((s) => s.inv)
  const setInv = useResearch((s) => s.setInv)
  const fr = inv.fisher.result
  const [which, setWhich] = useState(-1)
  const cv = useChannelValue(which)
  const picks = useMemo(() => new Map<number, number>(), [])
  const byEx = useMemo(() => {
    if (!cv || !fr) return []
    const m = new Map<number, number>()
    fr.channels.ex_nm.forEach((e, c) => m.set(e, (m.get(e) ?? 0) + cv.v[c]))
    const tot = [...m.values()].reduce((a, b) => a + b, 0) || 1
    return [...m.entries()].sort((a, b) => a[0] - b[0]).map(([e, v]) => ({ e, share: v / tot }))
  }, [cv, fr])
  const top = useMemo(() => (cv && fr ? [...cv.v.keys()].sort((a, b) => cv.v[b] - cv.v[a]).slice(0, 6).map((c) => ({ c, share: cv.v[c] / (cv.v.reduce((a, b) => a + b, 0) || 1) })) : []), [cv, fr])
  const running = inv.fisher.state === 'running'
  return (
    <div className="lf-card">
      <header><Grid3x3 size={14} style={{ color: 'var(--accent)' }} />Fisher information and Cramér–Rao bounds <small>how well can each parameter be measured, per channel</small><span className="sp" />
        {running ? <button className="btn sm danger" onClick={() => cancelInverseJob(currentJob())}><Square size={12} />Cancel</button> : <button className="btn sm primary" disabled={inv.params.length === 0} onClick={() => void runFisher()}><Play size={12} />Compute</button>}</header>
      <div className="cbody">
        <RunStatus b={inv.fisher} label="finite-difference Jacobian…" />
        {!fr && inv.fisher.state === 'idle' && <div className="hint" style={{ padding: '10px 0 4px' }}>The simulator is differentiated at the centre of each parameter range (2 P + 1 runs). With the Poisson–Gaussian noise model of the camera this gives the Fisher information of every (excitation, emission band) channel and the Cramér–Rao lower bound on the standard deviation of any unbiased estimate.</div>}
        {fr && (
          <>
            <table className="stat rs-crlb"><thead><tr><th>parameter</th><th>θ₀</th><th>CRLB sd</th><th>relative</th><th>identifiable</th></tr></thead>
              <tbody>{fr.names.map((n, p) => <tr key={n}><td>{n}</td><td>{fmtVal(fr.theta[p])}</td><td>{fmtSd(fr.crlb_sd[p])}</td><td>{fr.crlb_rel_sd[p] != null && Number.isFinite(fr.crlb_rel_sd[p]!) ? `${(fr.crlb_rel_sd[p]! * 100).toFixed(2)} %` : '∞'}</td><td>{Number.isFinite(fr.crlb_sd[p] ?? NaN) ? <span style={{ color: C_OK }}>yes</span> : <span style={{ color: C_NOISE }}>no</span>}</td></tr>)}</tbody></table>
            {!fr.identifiable && <div className="note warn" style={{ margin: '8px 0 0' }}>The parameters are not jointly identifiable from these channels (singular Fisher matrix): at least one direction has an infinite bound. Add excitations / bands or drop a parameter.</div>}
            <div className="rs-ctrl"><span className="hint">information about</span>
              <select className="field" aria-label="Parameter shown in the map" value={which} onChange={(e) => setWhich(Number(e.target.value))} style={{ width: 'auto' }}>
                <option value={-1}>all parameters (share of each)</option>{fr.names.map((n, i) => <option key={n} value={i}>{n}</option>)}</select>
              <span className="hint" style={{ marginLeft: 'auto' }}>{fr.n_channels} channels</span></div>
            {cv && <ChannelMap z={cv.grid.z} xs={cv.grid.xs} ys={cv.grid.ys} index={cv.grid.index} picks={picks} height={Math.max(180, 26 * cv.grid.ys.length + 52)} unit={which < 0 ? 'share' : 'e⁻²/u²'} />}
            <div className="rs-two" style={{ alignItems: 'start', paddingTop: 8 }}>
              <div><div className="hint" style={{ paddingBottom: 4 }}>Information by excitation</div>
                <div className="rs-bars">{byEx.map((b) => <div key={b.e}><span className="mono" style={{ color: wavelengthToHex(b.e) }}>{b.e}</span><div className="bar"><i style={{ width: `${Math.max(1, b.share * 100)}%`, background: wavelengthToHex(b.e) }} /></div><span className="mono">{(b.share * 100).toFixed(0)} %</span></div>)}</div></div>
              <div><div className="hint" style={{ paddingBottom: 4 }}>Most informative channels</div>
                <div className="rs-bars">{top.map((t) => <div key={t.c}><span className="mono" style={{ whiteSpace: 'nowrap' }}>{fr.channels.ex_nm[t.c]}→{Math.round(fr.channels.em_nm[t.c])}</span><div className="bar"><i style={{ width: `${Math.max(1, t.share * 400)}%`, background: C_S1 }} /></div><span className="mono">{(t.share * 100).toFixed(1)} %</span></div>)}</div></div>
            </div>
          </>)}
        <input type="hidden" value={inv.k} readOnly onChange={() => setInv({})} />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------------------------- optimal design
function DesignPanel() {
  const inv = useResearch((s) => s.inv)
  const setInv = useResearch((s) => s.setInv)
  const fr = inv.fisher.result, d = inv.design.result
  const [pShow, setPShow] = useState(0)
  const kmax = fr ? Math.min(60, fr.n_channels) : 20
  const cv = useChannelValue(-1)
  const picks = useMemo(() => { const m = new Map<number, number>(); d?.selected.forEach((c, i) => m.set(c, i + 1)); return m }, [d])
  const p = Math.min(pShow, (fr?.names.length ?? 1) - 1)
  const base = useMemo(() => (fr && d ? designBaselines(fr, p, d.k) : null), [fr, d, p])
  const curve = d?.crlb_sd_vs_k.map((row) => row[p] ?? NaN)
  const allSd = d?.all_channels_sd[p] ?? NaN
  const fmtT = (v: number) => (v >= 10 ? `${v.toFixed(1)} s` : v >= 0.1 ? `${v.toFixed(2)} s` : `${(v * 1000).toFixed(0)} ms`)
  return (
    <div className="lf-card">
      <header><Target size={14} style={{ color: 'var(--accent)' }} />Optimal channel design <small>which k (excitation, band) channels to acquire</small><span className="sp" />
        <button className="btn sm primary" disabled={!fr || inv.design.state === 'running'} onClick={() => void runDesign()}><Play size={12} />Select channels</button></header>
      <div className="cbody">
        {!fr && <div className="hint" style={{ padding: '10px 0' }}>Compute the Fisher information first (previous tab): the design selects the channels that make the Cramér–Rao bound smallest.</div>}
        {fr && (
          <>
            <div className="rs-opts">
              <div style={{ gridColumn: '1 / -1' }}><div className="hint">channels to keep <b style={{ color: 'var(--text)', fontFamily: 'var(--mono)' }}>k = {Math.min(inv.k, kmax)}</b> of {fr.n_channels}</div>
                <input type="range" min={1} max={kmax} value={Math.min(inv.k, kmax)} aria-label="Number of channels k" onChange={(e) => setInv({ k: Number(e.target.value) })} /></div>
              <div><div className="hint">criterion</div>
                <Seg small value={inv.criterion === 'D' || inv.criterion === 'A' ? inv.criterion : 'target:'} onChange={(v) => setInv({ criterion: v === 'target:' ? `target:${fr.names[inv.targetParam] ?? fr.names[0]}` : v })} options={[{ value: 'D', label: 'D-opt', title: 'Minimise the volume of the joint confidence ellipsoid (det of the CRLB covariance)' }, { value: 'A', label: 'A-opt', title: 'Minimise the sum of the parameter variances' }, { value: 'target:', label: 'Target', title: 'Minimise the variance of one parameter' }]} /></div>
              {inv.criterion.startsWith('target:') && <div><div className="hint">target parameter</div><Sel value={inv.targetParam} onChange={(v) => setInv({ targetParam: v, criterion: `target:${fr.names[v]}` })} options={fr.names.map((n, i) => ({ value: i, label: n }))} /></div>}
              <div><div className="hint">budget</div><Seg small value={inv.budget} onChange={(v) => setInv({ budget: v })} options={[{ value: 'channels', label: 'Channels' }, { value: 'exposure_time', label: 'Exposure time' }]} /></div>
              {inv.budget === 'exposure_time' && <div><div className="hint">total time</div><Num value={inv.totalTimeS} min={0.1} max={3600} step={0.5} bar={false} unit="s" onChange={(v) => setInv({ totalTimeS: v })} /></div>}
            </div>
            <RunStatus b={inv.design} label="selecting…" />
            {cv && <ChannelMap z={cv.grid.z} xs={cv.grid.xs} ys={cv.grid.ys} index={cv.grid.index} picks={picks} height={Math.max(180, 26 * cv.grid.ys.length + 52)} unit="share" />}
            {d && <div className="rs-picks" aria-label="Selected channels">{d.labels.map((l, i) => <span key={l} className="rs-pick"><b>{i + 1}</b>{l.replace('->', ' → ')}</span>)}</div>}
          </>)}
        {d && fr && base && curve && (
          <>
            <div className="rs-ctrl" style={{ paddingTop: 12 }}><span className="hint">CRLB sd of</span>
              <select className="field" style={{ width: 'auto' }} aria-label="Parameter of the CRLB curve" value={p} onChange={(e) => setPShow(Number(e.target.value))}>{fr.names.map((n, i) => <option key={n} value={i}>{n}</option>)}</select>
              <span className="hint" style={{ marginLeft: 'auto' }}>against the number of channels</span></div>
            <XYChart height={230} logY zeroBase={false} xLabel="channels k" yLabel={`CRLB sd (${useResearch.getState().inv.params[p]?.unit || 'units'})`} legend
              xDomain={[1, Math.max(2, d.k)]}
              series={[
                { name: 'random subset, 10–90 %', x: base.ks, y: base.random.mean.map((_, i) => Math.sqrt(Math.max(1e-300, base.random.lo[i] * base.random.hi[i]))), std: base.random.mean.map((_, i) => (base.random.hi[i] - base.random.lo[i]) / 2 * 0.5), color: '#7b8798', width: 1.4, opacity: 0.9 },
                { name: 'uniform spacing', x: base.ks, y: base.uniform, color: C_NOISE, width: 1.8, dash: '5 3' },
                { name: 'greedy + exchange design', x: curve.map((_, i) => i + 1), y: curve, color: C_S1, width: 2.6, points: true },
                { name: 'all channels', x: [1, Math.max(2, d.k)], y: [allSd, allSd], color: C_OK, width: 1.4, dash: '2 4' },
              ]} />
            <div className="rs-facts">
              <span>design <b style={{ color: C_S1 }}>{fmtSd(curve[curve.length - 1])}</b></span>
              <span>random (median) <b>{fmtSd(base.random.mean[base.random.mean.length - 1])}</b></span>
              <span>uniform <b>{fmtSd(base.uniform[base.uniform.length - 1])}</b></span>
              <span>all {fr.n_channels} channels <b style={{ color: C_OK }}>{fmtSd(allSd)}</b></span>
              {d.exhaustive && <Tip text={`Exhaustive search over all ${d.exhaustive.n_combinations.toLocaleString()} subsets: the greedy+exchange set ${d.exhaustive.same_set ? 'is the global optimum' : `is within ${d.exhaustive.refined_gap.toExponential(1)} of it in the criterion`}.`}><span style={{ color: d.exhaustive.same_set ? C_OK : C_NOISE }}>{d.exhaustive.same_set ? 'matches the exhaustive optimum' : 'near-optimal (exhaustive check)'}</span></Tip>}
            </div>
            {d.exposure && (
              <div style={{ paddingTop: 10 }}>
                <div className="hint" style={{ paddingBottom: 4 }}>Exposure allocation for {fmtT(d.exposure.total_time_s)} in total (CRLB sd of {fr.names[p]}: optimised <b style={{ color: C_S1 }}>{fmtSd(d.exposure.crlb_sd_opt[p])}</b>, equal split <b>{fmtSd(d.exposure.crlb_sd_equal[p])}</b>)</div>
                <div className="rs-bars wide">{d.exposure.channels.ex_nm.map((e, i) => { const mx = Math.max(...d.exposure!.t_opt_s); return <div key={i}><span className="mono" style={{ whiteSpace: 'nowrap' }}>{e}→{Math.round(d.exposure!.channels.em_nm[i])}</span><div className="bar"><i style={{ width: `${(d.exposure!.t_opt_s[i] / mx) * 100}%`, background: wavelengthToHex(e) }} /></div><span className="mono">{fmtT(d.exposure!.t_opt_s[i])}</span></div> })}</div>
              </div>)}
          </>)}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------------------------- fits
function FitPanel() {
  const inv = useResearch((s) => s.inv)
  const setInv = useResearch((s) => s.setInv)
  const captures = useStudio((s) => s.captures)
  const sur = inv.surrogate.result, fit = inv.fit.result, fm = inv.fitMap.result
  const cap = captures.find((c) => c.id === (inv.captureId ?? defaultCapture()))
  const gtNames = (cap?.ground_truth_layers ?? []).filter((n) => n.startsWith('layer_thickness'))
  const [gt, setGt] = useState('')
  const gtSel = gt || gtNames[0] || ''
  const [pp, setPp] = useState(0)
  const prof = fit?.profile ? fit.profile[fit.names[Math.min(pp, fit.names.length - 1)]] : null
  const running = inv.surrogate.state === 'running'
  const grid2 = (a: unknown): number[] => (Array.isArray(a) ? (a as unknown[]).flatMap((r) => (Array.isArray(r) ? (r as (number | null)[]).map((v) => (v == null ? NaN : v)) : [])) : [])
  const mapParam = (th: unknown): { w: number; h: number; d: Float32Array } | null => {
    if (!fm) return null
    const [H, W] = fm.shape
    const out = new Float32Array(H * W).fill(NaN)
    const t = th as unknown[][]
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) { const cell = t[r]?.[c]; const v = Array.isArray(cell) ? cell[0] : cell; out[r * W + c] = typeof v === 'number' ? v : NaN }
    return { w: W, h: H, d: out }
  }
  const est = mapParam(fm?.theta), gtm = fm?.gt ? { w: fm.shape[1], h: fm.shape[0], d: Float32Array.from(grid2(fm.gt)) } : null
  const err = est && gtm ? { w: est.w, h: est.h, d: Float32Array.from(est.d, (v, i) => v - gtm.d[i]) } : null
  const stats = (fm?.stats ?? {}) as Record<string, number | null>
  const st = typeof Object.values(stats)[0] === 'object' ? (Object.values(stats)[0] as unknown as Record<string, number | null>) : stats
  const lim = err ? Math.max(1e-9, ...Array.from(err.d, (v) => (Number.isFinite(v) ? Math.abs(v) : 0))) : 1
  return (
    <div className="lf-card">
      <header><Crosshair size={14} style={{ color: 'var(--accent)' }} />Parameter estimation <small>surrogate model of the simulator → fit the observed spectra</small><span className="sp" /></header>
      <div className="cbody">
        <div className="rs-step"><span className="n">1</span><div style={{ flex: 1 }}>
          <b>Build the surrogate</b><div className="hint">Simulates the scene on a space-filling design of the parameter ranges and interpolates every channel.</div>
          <div className="rs-ctrl"><span className="hint">simulator runs</span><div style={{ width: 84 }}><Num value={inv.gridPoints} min={3} max={128} integer bar={false} onChange={(v) => setInv({ gridPoints: v })} /></div>
            {running ? <button className="btn sm danger" onClick={() => cancelInverseJob(currentJob())}><Square size={12} />Cancel</button> : <button className="btn sm primary" disabled={inv.params.length === 0} onClick={() => void buildSurrogate()}><Play size={12} />Build</button>}</div>
          <RunStatus b={inv.surrogate} label="simulating…" />
          {sur && <div className="rs-facts"><span>{sur.names.join(', ')}</span><span>{sur.n_points} points</span><span>{sur.n_channels} channels</span>{sur.loo && <Tip text="Leave-one-out relative RMS error of the interpolation: how well the surrogate predicts a simulator run it has not seen."><span>LOO error <b>{(sur.loo.loo_rel_rms_mean * 100).toFixed(2)} %</b> (max {(sur.loo.loo_rel_rms_max * 100).toFixed(2)} %)</span></Tip>}</div>}
        </div></div>
        <div className="rs-step"><span className="n">2</span><div style={{ flex: 1 }}>
          <b>Fit the ROI</b><div className="hint">Maximum-likelihood estimate from the ROI-mean spectrum of the selected capture, with profile-likelihood confidence intervals.</div>
          <div className="rs-ctrl"><button className="btn sm primary" disabled={!sur || inv.fit.state === 'running'} onClick={() => void runFit()}><Play size={12} />Fit ROI</button></div>
          <RunStatus b={inv.fit} label="fitting…" />
          {fit && (
            <>
              <table className="stat rs-crlb"><thead><tr><th>parameter</th><th>estimate</th><th>± se</th><th>95 % CI</th><th /></tr></thead>
                <tbody>{fit.names.map((n, i) => { const ci = fit.ci[n]; return <tr key={n}><td>{n}</td><td><b style={{ color: C_S1 }}>{fmtVal(fit.theta[i])}</b></td><td>{fmtSd(fit.se[i])}</td><td>{ci ? `[${fmtVal(ci.lo ?? NaN)}, ${fmtVal(ci.hi ?? NaN)}]` : '—'}</td><td>{fit.at_bound[i] ? <span style={{ color: C_NOISE }}>at bound</span> : ''}</td></tr> })}</tbody></table>
              <div className="rs-facts"><span>χ²/dof <b>{fit.reduced_chi2.toFixed(2)}</b></span><span>dof {fit.dof}</span>{fit.n_roi_pixels != null && <span>{fit.n_roi_pixels} px</span>}</div>
              {prof && (
                <>
                  <div className="rs-ctrl"><span className="hint">profile likelihood</span><select className="field" style={{ width: 'auto' }} aria-label="Parameter of the profile" value={pp} onChange={(e) => setPp(Number(e.target.value))}>{fit.names.map((n, i) => <option key={n} value={i}>{n}</option>)}</select></div>
                  <XYChart height={150} zeroBase xLabel={fit.names[pp]} yLabel="Δχ²" series={[{ name: 'Δχ²', x: prof.values, y: prof.delta_chi2, color: C_ST, width: 2.2, fill: true }, { name: '95 % (3.84)', x: [prof.values[0], prof.values[prof.values.length - 1]], y: [3.84, 3.84], color: C_NOISE, dash: '4 3', width: 1.3 }]} yDomain={[0, Math.max(8, Math.min(30, Math.max(...prof.delta_chi2.filter(Number.isFinite))))]} legend />
                </>)}
            </>)}
        </div></div>
        <div className="rs-step"><span className="n">3</span><div style={{ flex: 1 }}>
          <b>Retrieve a map</b><div className="hint">Fit every tile of the image and compare with the simulator’s ground truth.</div>
          <div className="rs-ctrl"><span className="hint">tile</span><div style={{ width: 70 }}><Num value={inv.tile} min={1} max={64} integer bar={false} unit="px" onChange={(v) => setInv({ tile: v })} /></div>
            {gtNames.length > 0 && <select className="field" style={{ width: 'auto', maxWidth: 190 }} aria-label="Ground truth" value={gtSel} onChange={(e) => setGt(e.target.value)}>{gtNames.map((g) => <option key={g} value={g}>{g}</option>)}</select>}
            <button className="btn sm primary" disabled={!sur || inv.fitMap.state === 'running'} onClick={() => void runFitMap(gtSel || null)}><Play size={12} />Retrieve</button></div>
          <RunStatus b={inv.fitMap} label="retrieving…" />
          {fm && est && (
            <>
              <div className="rs-maps">
                <figure className="rs-map"><figcaption><i style={{ background: C_S1 }} />estimate<span>{fm.names[0]}</span></figcaption><MapCanvas data={est.d} w={est.w} h={est.h} cmap="viridis" maxHeight={150} label="estimated parameter map" vmin={gtm ? Math.min(...Array.from(gtm.d).filter(Number.isFinite)) : undefined} vmax={gtm ? Math.max(...Array.from(gtm.d).filter(Number.isFinite)) : undefined} /></figure>
                {gtm && <figure className="rs-map"><figcaption><i style={{ background: C_OK }} />ground truth<span>{gtSel}</span></figcaption><MapCanvas data={gtm.d} w={gtm.w} h={gtm.h} cmap="viridis" maxHeight={150} label="ground-truth map" /></figure>}
                {err && <figure className="rs-map"><figcaption><i style={{ background: C_NOISE }} />error<span>estimate − truth</span></figcaption><MapCanvas data={err.d} w={err.w} h={err.h} cmap="turbo" vmin={-lim} vmax={lim} maxHeight={150} label="estimation error" /></figure>}
              </div>
              <div className="rs-facts">{st.bias != null && <span>bias <b>{fmtVal(st.bias)}</b></span>}{st.rmse != null && <span>RMSE <b>{fmtVal(st.rmse)}</b></span>}{st.pearson_r != null && <span>r <b>{st.pearson_r.toFixed(3)}</b></span>}{st.coverage_1p96se != null && <span>coverage (±1.96 se) <b>{(st.coverage_1p96se * 100).toFixed(0)} %</b></span>}</div>
            </>)}
        </div></div>
      </div>
    </div>
  )
}

export function InverseView() {
  const tab = useResearch((s) => s.inv.tab)
  const setInv = useResearch((s) => s.setInv)
  return (
    <div className="sw rs">
      <Setup />
      <div className="res">
        <div className="rs-subtabs" role="tablist" aria-label="Inverse tools">
          {([['info', 'Information', Grid3x3], ['design', 'Design', Target], ['fit', 'Estimation', Crosshair], ['unmix', 'Unmixing', Atom]] as const).map(([id, label, Icon]) => (
            <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? 'on' : ''} onClick={() => setInv({ tab: id })}><Icon size={13} />{label}</button>))}
          <span className="sp" /><Gauge size={12} style={{ color: 'var(--faint)' }} /><span className="hint">Poisson–Gaussian noise, in electrons</span>
        </div>
        {tab === 'unmix' && <UnmixPanel />}
        {tab === 'info' && <InfoPanel />}
        {tab === 'design' && <DesignPanel />}
        {tab === 'fit' && <FitPanel />}
      </div>
    </div>
  )
}

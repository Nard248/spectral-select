import { useEffect, useState } from 'react'
import { Play, Camera, Moon, Square as Stop, CircleDot, AlertTriangle, Info, Plus, Eye } from 'lucide-react'
import { useStudio, activeSource } from '../state/store'
import { runPreview, runCapture, cancelJob, estimateSeconds } from '../state/actions'
import { wavelengthToHex, isInvisible } from '../lib/color'
import { Tip } from '../ui/controls'
import { etaSeconds, fmtEta } from '../state/progress'

const fmtTime = (s: number) => (s < 90 ? `${s.toFixed(0)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`)

export function AcquireBar() {
  const exp = useStudio((s) => s.exp)
  const src = activeSource(exp)
  const job = useStudio((s) => s.job)
  const previewing = useStudio((s) => s.previewing)
  const previewNm = useStudio((s) => s.previewNm)
  const warnings = useStudio((s) => s.warnings)
  const errors = useStudio((s) => s.errors)
  const edit = useStudio((s) => s.edit)
  const select = useStudio((s) => s.select)
  const [open, setOpen] = useState(false)
  const running = job?.state === 'running'
  const [, tick] = useState(0)
  useEffect(() => { if (!running) return; const t = setInterval(() => tick((n) => n + 1), 500); return () => clearInterval(t) }, [running])
  const eta = job && running ? etaSeconds(job.fraction, job.startedAt) : null
  const eff = job?.activeNm ?? previewNm
  const secs = estimateSeconds(exp)
  const nWarn = warnings.filter((w) => w.severity === 'warn').length

  return (
    <div className="bottombar" style={{ position: 'relative' }}>
      {open && (warnings.length > 0 || errors.length > 0) && (
        <div className="glass" style={{ position: 'absolute', left: 12, bottom: '100%', marginBottom: 8, width: 520, maxWidth: 'calc(100% - 24px)', padding: 10, zIndex: 30, display: 'grid', gap: 6, maxHeight: 260, overflow: 'auto' }}>
          {errors.map((e, i) => <div key={`e${i}`} className="status"><span className="w err"><AlertTriangle size={13} />{e.loc.join('.')}: {e.msg}</span></div>)}
          {warnings.map((w, i) => <div key={i} style={{ display: 'flex', gap: 8, color: w.severity === 'warn' ? 'var(--warn)' : 'var(--dim)', fontSize: 12 }}>{w.severity === 'warn' ? <AlertTriangle size={13} style={{ flex: 'none', marginTop: 2 }} /> : <Info size={13} style={{ flex: 'none', marginTop: 2 }} />}<span>{w.message}</span></div>)}
        </div>
      )}
      <div className="seq">
        <Tip title="Excitation sequence" text="Order matters — photobleaching accumulates step by step. Click a chip to preview that line; double-click toggles it; edit details in the Source inspector."><span className="ptitle" style={{ padding: 0, whiteSpace: 'nowrap' }}>Sequence</span></Tip>
        <div className="chips">
          {src.sequence.map((st, i) => {
            const c = wavelengthToHex(st.wavelength_nm)
            const active = running && job?.activeNm === st.wavelength_nm
            return (
              <div key={i} className={`exchip${st.enabled ? '' : ' off'}${isInvisible(st.wavelength_nm) ? ' uv' : ''}${active ? ' active' : ''}${eff === st.wavelength_nm && !running ? ' sel' : ''}`}
                style={{ ['--c' as string]: c }} role="button" tabIndex={0}
                title={`${st.wavelength_nm} nm · ${st.power_mW} mW · ${st.exposure_ms} ms${isInvisible(st.wavelength_nm) ? ' (invisible)' : ''}`}
                onClick={() => { useStudio.setState({ previewNm: st.wavelength_nm }); select({ kind: 'source', id: src.id }) }}
                onDoubleClick={() => edit((e) => { const s = e.sources.find((x) => x.id === src.id); if (s) s.sequence[i].enabled = !s.sequence[i].enabled })}>
                <span className="num-badge">{i + 1}</span>
                <span className="wl">{st.wavelength_nm}<span style={{ fontSize: 9.5, opacity: 0.7 }}> nm</span></span>
                <small>{st.power_mW} mW · {st.exposure_ms} ms</small>
              </div>)
          })}
          <button className="btn ghost icon" title="Add excitation step" style={{ alignSelf: 'center' }} onClick={() => {
            edit((e) => { const s = e.sources.find((x) => x.id === src.id); if (s) { const last = s.sequence.at(-1); s.sequence.push({ wavelength_nm: Math.min(1090, (last?.wavelength_nm ?? 385) + 20), power_mW: last?.power_mW ?? 3, exposure_ms: last?.exposure_ms ?? 100, enabled: true }) } })
          }}><Plus size={15} /></button>
        </div>
        <span className="hint mono" style={{ whiteSpace: 'nowrap' }} title="Estimated acquisition time (exposure × bands + filter retune)">≈ {fmtTime(secs)}</span>
      </div>
      <div className="status" style={{ gap: 12 }}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flex: 1, minWidth: 0 }}>
        {job ? <>
          <span className="w" style={{ color: job.state === 'error' ? 'var(--err)' : 'var(--accent)' }}><CircleDot size={12} />{job.label ?? job.message}</span>
          <div className="progress"><i style={{ width: `${Math.round(job.fraction * 100)}%` }} /></div>
          <span className="mono">{Math.round(job.fraction * 100)}%</span>
          {running && <span className="mono hint" style={{ whiteSpace: 'nowrap' }}>{eta != null ? fmtEta(eta) : 'estimating…'}</span>}
        </> : <>
          <span style={{ cursor: warnings.length || errors.length ? 'pointer' : 'default', display: 'inline-flex', gap: 10, alignItems: 'center' }} onClick={() => setOpen(!open)}>
            {errors.length > 0 && <span className="w err"><AlertTriangle size={12} />{errors.length} error{errors.length > 1 ? 's' : ''}</span>}
            {nWarn > 0 && <span className="w warn"><AlertTriangle size={12} />{nWarn} warning{nWarn > 1 ? 's' : ''}</span>}
            {warnings.length - nWarn > 0 && <span className="w info"><Info size={12} />{warnings.length - nWarn} note{warnings.length - nWarn > 1 ? 's' : ''}</span>}
            {!errors.length && !warnings.length && <span className="w">Ready · no issues</span>}
          </span>
          <span style={{ marginLeft: 'auto' }} className="hint">Camera {exp.cameras.find((c) => c.id === exp.active_camera_id)?.sensor.width_px ?? ''}×{exp.cameras.find((c) => c.id === exp.active_camera_id)?.sensor.height_px ?? ''} · seed {exp.acquisition.seed}</span>
        </>}
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flex: 'none' }}>
        <Tip text="Fast low-res engine render of one excitation  [Space]"><button className="btn" disabled={running || previewing} onClick={() => void runPreview()}><Eye size={14} />{previewing ? 'Previewing…' : 'Preview'}</button></Tip>
        {running
          ? <button className="btn danger" onClick={() => void cancelJob()}><Stop size={13} />Cancel</button>
          : <button className="btn primary" disabled={previewing} onClick={() => void runCapture('sample')}><Play size={13} fill="currentColor" />Capture</button>}
        <Tip text="Dark frame: shutter closed — offset, dark current, read noise"><button className="btn" disabled={running} onClick={() => void runCapture('dark')}><Moon size={14} />Dark</button></Tip>
        <Tip text="White reference: uniform diffuse target for flat-fielding"><button className="btn" disabled={running} onClick={() => void runCapture('white_reference')}><Camera size={14} />White ref</button></Tip>
        </div>
      </div>
    </div>
  )
}

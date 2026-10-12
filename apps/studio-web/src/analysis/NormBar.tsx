import { AlertTriangle } from 'lucide-react'
import { useEffect, useMemo } from 'react'
import type { CaptureMeta } from '../api/types'
import { Sel, Seg, Tip } from '../ui/controls'
import { useAnalysis } from './store'
import { NORM_MODES, compatibility, normProblems, normUnit } from './norm'

/** Normalisation + statistic selector with dark / white capture pickers that validate compatibility. */
export function NormBar({ meta, all }: { meta: CaptureMeta; all: CaptureMeta[] }) {
  const norm = useAnalysis((s) => s.norm)
  const setNorm = useAnalysis((s) => s.setNorm)
  const stat = useAnalysis((s) => s.stat)
  const setStat = useAnalysis((s) => s.setStat)
  const needDark = norm.mode === 'dark_subtracted' || norm.mode === 'white_normalized'
  const needWhite = norm.mode === 'white_normalized'

  const darks = useMemo(() => all.filter((c) => c.mode === 'dark' && c.id !== meta.id), [all, meta.id])
  const whites = useMemo(() => all.filter((c) => c.mode === 'white_reference' && c.id !== meta.id), [all, meta.id])
  const problems = useMemo(() => normProblems(norm, meta, all), [norm, meta, all])

  // auto-pick the newest compatible reference when a mode needing one is selected
  useEffect(() => {
    if (needDark && !norm.dark_capture_id) {
      const d = darks.find((c) => compatibility(meta, c, 'dark').length === 0) ?? darks[0]
      if (d) setNorm({ dark_capture_id: d.id })
    }
    if (needWhite && !norm.white_capture_id) {
      const w = whites.find((c) => compatibility(meta, c, 'white').length === 0) ?? whites[0]
      if (w) setNorm({ white_capture_id: w.id })
    }
  }, [needDark, needWhite, norm.dark_capture_id, norm.white_capture_id, darks, whites, meta, setNorm])

  const label = (c: CaptureMeta, role: 'dark' | 'white') => `${c.name} · ${new Date(c.created).toLocaleTimeString()}${compatibility(meta, c, role).length ? ' (incompatible)' : ''}`
  return (
    <div className="normbar">
      <div className="r"><label>Units</label>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <div style={{ flex: 1, minWidth: 0 }}><Sel value={norm.mode} options={NORM_MODES.map((m) => ({ value: m.value, label: m.label }))} onChange={(mode) => setNorm({ mode })} /></div>
          <Tip text="Statistic over the ROI pixels per band."><span><Seg small value={stat} onChange={setStat} options={[{ value: 'mean', label: 'mean' }, { value: 'median', label: 'median' }]} /></span></Tip>
          <span className="unit" title="unit of every axis in the analysis panel">{normUnit(norm.mode)}</span>
        </div>
      </div>
      <div className="hint" style={{ margin: '-2px 0 0 70px' }} title={NORM_MODES.find((m) => m.value === norm.mode)?.tip}>{NORM_MODES.find((m) => m.value === norm.mode)?.tip}</div>
      {(needDark || needWhite) && <div className="r"><label>Refs</label>
        <div className="pair">
          {needDark && <Sel value={norm.dark_capture_id ?? ''} options={[{ value: '', label: darks.length ? '— dark capture —' : '— no dark capture —' }, ...darks.map((c) => ({ value: c.id, label: label(c, 'dark') }))]} onChange={(v) => setNorm({ dark_capture_id: v || null })} />}
          {needWhite && <Sel value={norm.white_capture_id ?? ''} options={[{ value: '', label: whites.length ? '— white reference —' : '— no white capture —' }, ...whites.map((c) => ({ value: c.id, label: label(c, 'white') }))]} onChange={(v) => setNorm({ white_capture_id: v || null })} />}
        </div></div>}
      {problems.length > 0 && <div className="bad"><AlertTriangle size={13} style={{ flex: 'none', marginTop: 2 }} /><span>{problems.join(' · ')}{(needDark && !darks.length) || (needWhite && !whites.length) ? ' — take a Dark / White ref capture in the Acquire bar.' : ''}</span></div>}
    </div>
  )
}

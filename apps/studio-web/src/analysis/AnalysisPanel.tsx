import './analysis.css'
import { useEffect } from 'react'
import { Grid3x3, MousePointerClick, Ruler, TrendingUp, Maximize2, Minimize2 } from 'lucide-react'
import type { CaptureMeta } from '../api/types'
import { useAnalysis, type AnalysisTab } from './store'
import { NormBar } from './NormBar'
import { EemPanel } from './EemPanel'
import { LinePanel } from './LinePanel'
import { ResponsePanel } from './ResponsePanel'
import { PixelPanel } from '../explorer/PixelPanel'
import { useStudio } from '../state/store'
import { useSweep } from '../sweep/store'
import { layeredMetric } from './layeredDefaults'

/** Captures already given the layered-block defaults (so the user's later choices are never overridden). */
const defaulted = new Set<string>()

const TABS: { id: AnalysisTab; label: string; icon: React.ReactNode }[] = [
  { id: 'pixel', label: 'Pixel', icon: <MousePointerClick size={13} /> },
  { id: 'eem', label: 'EEM', icon: <Grid3x3 size={13} /> },
  { id: 'line', label: 'Profile', icon: <Ruler size={13} /> },
  { id: 'response', label: 'Response', icon: <TrendingUp size={13} /> },
]

export function AnalysisPanel({ meta, all, pixel, ex }: { meta: CaptureMeta; all: CaptureMeta[]; pixel: { x: number; y: number } | null; ex: number | null }) {
  const tab = useAnalysis((s) => s.tab)
  const setTab = useAnalysis((s) => s.setTab)
  const wide = useAnalysis((s) => s.wide)
  const setWide = useAnalysis((s) => s.setWide)
  const exp = useStudio((s) => s.exp)
  // layered-block captures: default to the buried fluorophore's ex, the ex+40 nm band and irradiance_normalized
  useEffect(() => {
    if (defaulted.has(meta.id)) return
    const m = layeredMetric(meta, exp)
    if (!m) return
    defaulted.add(meta.id)
    const a = useAnalysis.getState()
    if (a.norm.mode === 'adu') a.setNorm({ mode: 'irradiance_normalized' })
    a.setResponse({ metric: m })
    useSweep.getState().setCfg({ metric: m })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta.id])
  return (
    <div className="apanel">
      <div className="atabs" role="tablist">
        {TABS.map((t) => <button key={t.id} role="tab" aria-selected={tab === t.id} className={tab === t.id ? 'on' : ''} onClick={() => setTab(t.id)}>{t.icon}{t.label}</button>)}
        <span className="sp" />
        <button className="iconbtn" title={wide ? 'Narrow panel' : 'Wide panel'} aria-label={wide ? 'Narrow analysis panel' : 'Widen analysis panel'} onClick={() => setWide(!wide)}>{wide ? <Minimize2 size={14} /> : <Maximize2 size={14} />}</button>
      </div>
      {tab !== 'pixel' && <NormBar meta={meta} all={all} />}
      {tab === 'pixel' && (ex != null ? <PixelPanel meta={meta} pixel={pixel} ex={ex} /> : null)}
      {tab === 'eem' && <EemPanel meta={meta} all={all} ex={ex} />}
      {tab === 'line' && <LinePanel meta={meta} ex={ex} />}
      {tab === 'response' && <ResponsePanel meta={meta} ex={ex} />}
    </div>
  )
}

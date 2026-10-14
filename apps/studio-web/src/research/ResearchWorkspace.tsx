import './research.css'
import { useEffect } from 'react'
import { FlaskConical, FileText, Sigma, Layers3 } from 'lucide-react'
import { useResearch, type ResearchTab } from './store'
import { StudyBuilder } from './StudyBuilder'
import { StudyResults } from './StudyResults'
import { InverseView } from './InverseView'
import { MethodsView } from './MethodsView'
import { attachStudy, loadCalibrated, loadHistory } from './actions'
import { loadResearchApi } from './api'

const TABS: { id: ResearchTab; label: string; icon: React.ReactNode; hint: string }[] = [
  { id: 'study', label: 'Studies', icon: <Layers3 size={14} />, hint: 'Design of experiments, sensitivity analysis, results' },
  { id: 'inverse', label: 'Inverse & design', icon: <Sigma size={14} />, hint: 'Unmixing, Cramér–Rao bounds, optimal channels, parameter estimation' },
  { id: 'methods', label: 'Methods & provenance', icon: <FileText size={14} />, hint: 'Generated Methods paragraph, software and experiment hash' },
]

/** The v1.2 "Research" top-level workspace. */
export function ResearchWorkspace() {
  const tab = useResearch((s) => s.tab)
  const set = useResearch((s) => s.set)
  useEffect(() => {
    let alive = true
    void (async () => {
      await loadResearchApi()
      await loadHistory()
      void loadCalibrated()
      if (!alive || useResearch.getState().run) return
      let last: string | null = null
      try { last = localStorage.getItem('sf.research.study') } catch { /* private mode */ }
      if (last && useResearch.getState().history.some((h) => h.study_id === last)) void attachStudy(last)
    })()
    return () => { alive = false }
  }, [])
  return (
    <div className="research">
      <div className="wsbar rs-bar">
        <div className="rs-brand"><FlaskConical size={15} /><b>Research</b></div>
        <div className="atabs" role="tablist" aria-label="Research tools" style={{ border: 0, background: 'transparent', padding: 0 }}>
          {TABS.map((t) => <button key={t.id} role="tab" aria-selected={tab === t.id} title={t.hint} className={tab === t.id ? 'on' : ''} onClick={() => set({ tab: t.id })}>{t.icon}{t.label}</button>)}
        </div>
        <span className="grow" />
        <span className="hint">reproducible, seeded, every number traceable to an experiment hash</span>
      </div>
      {tab === 'study' && <div className="sw rs"><StudyBuilder /><div className="res"><StudyResults /></div></div>}
      {tab === 'inverse' && <InverseView />}
      {tab === 'methods' && <div className="rs-scroll"><MethodsView /></div>}
    </div>
  )
}

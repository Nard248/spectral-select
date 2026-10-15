import { useEffect } from 'react'
import { useStudio } from './state/store'
import { TopBar } from './panels/TopBar'
import { Outliner } from './panels/Outliner'
import { Library } from './panels/Library'
import { Inspector } from './panels/Inspector'
import { AcquireBar } from './panels/AcquireBar'
import { Viewport } from './scene/Viewport'
import { Explorer } from './explorer/Explorer'
import { ResearchWorkspace } from './research/ResearchWorkspace'
import { Toast } from './ui/toast'
import { Tour } from './tour/Tour'
import { initEngine, startValidation } from './state/actions'
import { restoreAutosave, startAutosave } from './state/io'
import { useHotkeys } from './state/hotkeys'

export default function App() {
  const mode = useStudio((s) => s.mode)
  const ui = useStudio((s) => s.ui)
  useHotkeys()

  useEffect(() => {
    if (new URLSearchParams(location.search).get('restore') === '1') restoreAutosave()
    const stopSave = startAutosave()
    let stopVal: (() => void) | null = null
    void initEngine().then(() => { stopVal = startValidation() })
    if (import.meta.env.DEV) (window as unknown as { studio: typeof useStudio }).studio = useStudio
    return () => { stopSave(); stopVal?.() }
  }, [])

  const showWorkspace = mode !== 'explore' && mode !== 'research'
  return (
    <div className="app">
      <TopBar />
      <div className={`workspace${!ui.leftOpen || mode === 'acquire' ? ' no-left' : ''}${!ui.rightOpen ? ' no-right' : ''}`} style={{ display: showWorkspace ? undefined : 'none' }}>
        <aside className="panel left"><div className="scroll"><Outliner /></div><div style={{ borderTop: '1px solid var(--line)', display: 'flex', flexDirection: 'column', minHeight: 0, flex: '0 0 44%' }}><Library /></div></aside>
        <div className="center"><Viewport /><AcquireBar /></div>
        <aside className="panel right"><Inspector /></aside>
      </div>
      {mode === 'explore' && <Explorer />}
      {mode === 'research' && <ResearchWorkspace />}
      <Tour />
      <Toast />
    </div>
  )
}

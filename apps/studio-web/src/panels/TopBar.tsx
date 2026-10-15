import { useRef } from 'react'
import { Undo2, Redo2, Save, FolderOpen, Sparkles, ChevronDown, PanelLeft, PanelRight, RotateCcw, FilePlus2, Route } from 'lucide-react'
import { useStudio } from '../state/store'
import { PRESET_ORDER } from '../state/presets'
import { downloadExperiment, hasAutosave, pickAndOpen, restoreAutosave } from '../state/io'
import { initEngine } from '../state/actions'
import { MenuButton, Seg, Text, Tip } from '../ui/controls'
import { toast } from '../ui/toast'
import { startTour } from '../tour/Tour'
import { useTour } from '../tour/tourStore'

export function Logo() {
  return (
    <svg className="logo" viewBox="0 0 32 32">
      <defs><linearGradient id="lg" x1="0" x2="1"><stop offset="0" stopColor="#8b5cff" /><stop offset=".35" stopColor="#2ea8ff" /><stop offset=".65" stopColor="#4ef28a" /><stop offset="1" stopColor="#ff9b4a" /></linearGradient></defs>
      <rect width="32" height="32" rx="8" fill="#0d1014" stroke="rgba(255,255,255,.14)" />
      <path d="M6 22 L11 10 L16 19 L21 7 L26 22" fill="none" stroke="url(#lg)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

export function TopBar() {
  const name = useStudio((s) => s.exp.name)
  const edit = useStudio((s) => s.edit)
  const mode = useStudio((s) => s.mode)
  const setMode = useStudio((s) => s.setMode)
  const undo = useStudio((s) => s.undo), redo = useStudio((s) => s.redo)
  const canUndo = useStudio((s) => s.past.length > 0), canRedo = useStudio((s) => s.future.length > 0)
  const loadPreset = useStudio((s) => s.loadPreset)
  const engine = useStudio((s) => s.engine)
  const ui = useStudio((s) => s.ui), setUi = useStudio((s) => s.setUi)
  const dirty = useStudio((s) => s.dirty)
  const busy = useRef(false)
  const tourOn = useTour((t) => t.active)

  return (
    <header className="topbar">
      <div className="brand"><Logo /><span><b>SpectralForge</b> <small>Studio</small></span></div>
      <div className="sep" />
      <Tip text="Experiment name"><div style={{ width: 'clamp(150px, 14vw, 230px)' }}><Text value={name} onChange={(v) => edit((e) => { e.name = v }, 'name')} /></div></Tip>
      {dirty && <span className="tag" title="Unsaved changes">edited</span>}
      <MenuButton className="btn ghost" icon={<Sparkles size={14} />} label={<>Presets <ChevronDown size={13} /></>}>
        {(close) => <>
          {PRESET_ORDER.map((p) => (
            <button key={p.key} className="mi" onClick={() => { loadPreset(p.key); close(); toast(`Loaded ${p.label}`) }}>
              <span><b style={{ fontWeight: 600 }}>{p.label}</b><small>{p.blurb}</small></span>
            </button>
          ))}
        </>}
      </MenuButton>
      <MenuButton className="btn ghost" icon={<FolderOpen size={14} />} label={<>File <ChevronDown size={13} /></>}>
        {(close) => <>
          <button className="mi" onClick={() => { close(); pickAndOpen() }}><FolderOpen size={14} />Open .sfx.json…</button>
          <button className="mi" onClick={() => { close(); downloadExperiment() }}><Save size={14} />Save .sfx.json <span className="kbd" style={{ marginLeft: 'auto' }}>⌘S</span></button>
          <hr />
          <button className="mi" onClick={() => { close(); loadPreset('generic_macro_bench') }}><FilePlus2 size={14} />New from “Generic macro bench”</button>
          <button className="mi" disabled={!hasAutosave()} onClick={() => { close(); toast(restoreAutosave() ? 'Restored autosave' : 'No autosave found') }}><RotateCcw size={14} />Restore autosave</button>
        </>}
      </MenuButton>
      <Tip text="Guided tour: a captioned walkthrough of the bench, a capture and the physics"><button className="btn ghost" onClick={() => startTour()} disabled={tourOn}><Route size={14} />Tour</button></Tip>
      <div className="sep" />
      <Tip text="Undo  [⌘Z]"><button className="btn ghost icon" disabled={!canUndo} onClick={undo}><Undo2 size={16} /></button></Tip>
      <Tip text="Redo  [⇧⌘Z]"><button className="btn ghost icon" disabled={!canRedo} onClick={redo}><Redo2 size={16} /></button></Tip>
      <div className="spacer" />
      <Seg accent value={mode} onChange={setMode} options={[
        { value: 'design', label: <>Design</>, title: 'Build the scene [1]' },
        { value: 'acquire', label: <>Acquire</>, title: 'Sequence, preview, capture [2]' },
        { value: 'explore', label: <>Explore</>, title: 'Inspect captured cubes [3]' },
        { value: 'research', label: <>Research</>, title: 'Studies, sensitivity analysis, inverse problems, Methods text [4]' },
      ]} />
      <div className="spacer" />
      {engine === 'mock' && (
        <Tip title="Engine offline" text="The Python engine at 127.0.0.1:8765 is unreachable. Preview and capture use synthetic client-side data. Click to retry.">
          <span className="badge mock" onClick={async () => { if (busy.current) return; busy.current = true; await initEngine(true); busy.current = false }}><i />engine offline — mock data</span>
        </Tip>
      )}
      {engine === 'online' && <span className="badge ok"><i />engine online</span>}
      <Tip text="Toggle left panel  [ [ ]"><button className={`btn ghost icon${ui.leftOpen ? ' on' : ''}`} onClick={() => setUi({ leftOpen: !ui.leftOpen })}><PanelLeft size={16} /></button></Tip>
      <Tip text="Toggle inspector  [ ] ]"><button className={`btn ghost icon${ui.rightOpen ? ' on' : ''}`} onClick={() => setUi({ rightOpen: !ui.rightOpen })}><PanelRight size={16} /></button></Tip>
    </header>
  )
}

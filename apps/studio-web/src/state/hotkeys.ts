import { useEffect } from 'react'
import { useStudio } from './store'
import { viewApi } from '../scene/viewApi'
import { downloadExperiment } from './io'
import { runPreview } from './actions'
import { useLayerView, targetBlockId, DEFAULT_SECTION } from './layerView'
import { setCutaway } from '../scene/blockView'

const typing = (t: EventTarget | null) => {
  const el = t as HTMLElement | null
  return !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable)
}

export function useHotkeys() {
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (typing(e.target)) return
      const st = useStudio.getState()
      const mod = e.metaKey || e.ctrlKey
      const k = e.key.toLowerCase()
      if (mod && k === 'z') { e.preventDefault(); if (e.shiftKey) st.redo(); else st.undo(); return }
      if (mod && k === 'y') { e.preventDefault(); st.redo(); return }
      if (mod && k === 'd') { e.preventDefault(); st.duplicateSelected(); return }
      if (mod && k === 's') { e.preventDefault(); downloadExperiment(); return }
      if (mod || e.altKey) return
      if ((st.mode === 'explore' || st.mode === 'research') && !['1', '2', '3', '4'].includes(k)) return   // Explorer owns the remaining keys
      if (k === 'g') st.setUi({ gizmo: 'translate' })
      else if (k === 'r') st.setUi({ gizmo: 'rotate' })
      else if (k === 's') st.setUi({ snap: !st.ui.snap })
      else if (k === 'f') viewApi.current?.frame()
      else if (k === 'c') st.setUi({ lookThrough: !st.ui.lookThrough })
      else if (k === 'x') { const id = targetBlockId(st.exp, st.selection); const o = id ? st.exp.objects.find((x) => x.id === id) : null; if (o) setCutaway(o, !(useLayerView.getState().sections[o.id] ?? DEFAULT_SECTION).on) }
      else if (k === 'delete' || k === 'backspace') { e.preventDefault(); st.deleteSelected() }
      else if (k === 'escape') { if (st.ui.lookThrough) st.setUi({ lookThrough: false }); else st.select(null) }
      else if (k === ' ') { e.preventDefault(); void runPreview() }
      else if (k === '1') st.setMode('design')
      else if (k === '2') st.setMode('acquire')
      else if (k === '3') st.setMode('explore')
      else if (k === '4') st.setMode('research')
      else if (k === '[') st.setUi({ leftOpen: !st.ui.leftOpen })
      else if (k === ']') st.setUi({ rightOpen: !st.ui.rightOpen })
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [])
}

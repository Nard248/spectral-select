import type { Experiment } from '../generated/experiment'
import { useStudio } from './store'
import { toast } from '../ui/toast'

const KEY = 'sfx.autosave.v1'

export function serialize(e: Experiment): string { return JSON.stringify(e, null, 2) }

/** Parse + sanity-check a `.sfx.json`. Throws with a readable message. */
export function parseExperiment(text: string): Experiment {
  let j: unknown
  try { j = JSON.parse(text) } catch { throw new Error('not valid JSON') }
  const e = j as Partial<Experiment>
  if (!e || typeof e !== 'object') throw new Error('not an experiment object')
  for (const k of ['objects', 'sources', 'cameras', 'materials'] as const)
    if (!Array.isArray(e[k])) throw new Error(`missing "${k}" array`)
  if (!e.cameras!.length || !e.sources!.length) throw new Error('experiment needs at least one source and one camera')
  return { fluorophores: [], description: '', schema_version: '1.0', name: 'Untitled experiment', active_source_id: e.sources![0].id, active_camera_id: e.cameras![0].id, ...e } as Experiment
}

export function downloadExperiment() {
  const { exp } = useStudio.getState()
  const blob = new Blob([serialize(exp)], { type: 'application/json' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `${exp.name.replace(/[^\w-]+/g, '_') || 'experiment'}.sfx.json`
  document.body.appendChild(a); a.click(); a.remove()
  setTimeout(() => URL.revokeObjectURL(a.href), 2000)
  useStudio.setState({ dirty: false })
  toast('Saved ' + a.download)
}

export function openExperimentFile(file: File) {
  file.text().then((t) => {
    useStudio.getState().setExperiment(parseExperiment(t))
    toast(`Opened ${file.name}`)
  }).catch((err: Error) => toast(`Could not open ${file.name}: ${err.message}`, 'error'))
}

export function pickAndOpen() {
  const inp = document.createElement('input')
  inp.type = 'file'; inp.accept = '.json,.sfx.json,application/json'
  inp.onchange = () => { const f = inp.files?.[0]; if (f) openExperimentFile(f) }
  inp.click()
}

export function startAutosave() {
  let t: ReturnType<typeof setTimeout> | null = null
  return useStudio.subscribe((s, p) => {
    if (s.exp === p.exp) return
    if (t) clearTimeout(t)
    t = setTimeout(() => { try { localStorage.setItem(KEY, serialize(s.exp)) } catch { /* storage full / blocked */ } }, 600)
  })
}
export function restoreAutosave(): boolean {
  try {
    const t = localStorage.getItem(KEY); if (!t) return false
    useStudio.getState().setExperiment(parseExperiment(t)); return true
  } catch { return false }
}
export const hasAutosave = () => { try { return !!localStorage.getItem(KEY) } catch { return false } }

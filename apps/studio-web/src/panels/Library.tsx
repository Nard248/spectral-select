import { useState } from 'react'
import { Plus } from 'lucide-react'
import { useStudio } from '../state/store'
import { LIBRARY_SHAPES } from '../state/library'
import { BUNDLE, clone } from '../state/presets'
import { ShapeIcon, SourceIcon, CameraIcon } from '../ui/icons'
import { Seg, Tip } from '../ui/controls'
import { wavelengthToHex } from '../lib/color'
import { toast } from '../ui/toast'
import type { SceneObject } from '../generated/experiment'

const KIND: Record<string, SceneObject['shape']['kind']> = { petri: 'petri_dish', cuvette: 'cuvette', slide: 'box', droplet: 'droplet', bead: 'sphere', block: 'box', puck: 'cylinder' }

export function Library() {
  const [tab, setTab] = useState<'ware' | 'materials' | 'dyes' | 'hardware'>('ware')
  const addObject = useStudio((s) => s.addObject)
  const edit = useStudio((s) => s.edit)
  const sel = useStudio((s) => s.selection)
  const exp = useStudio((s) => s.exp)
  const select = useStudio((s) => s.select)
  const addSource = useStudio((s) => s.addSource), addCamera = useStudio((s) => s.addCamera)

  const assignMaterial = (id: string) => {
    const matId = id
    const preset = BUNDLE.materials.find((m) => m.id === id)!
    const existing = exp.materials.find((m) => m.id === id)
    if (!existing) edit((e) => { e.materials.push(clone(preset)) })
    if (sel?.kind === 'object') {
      edit((e) => { const o = e.objects.find((x) => x.id === sel.id); if (o) o.material_id = matId })
      toast(`${preset.name} → ${exp.objects.find((o) => o.id === sel.id)?.name}`)
    } else { select({ kind: 'material', id: matId }); toast(`${preset.name} added to experiment`) }
  }
  const addDye = (name: string) => {
    const matId: string | null = sel?.kind === 'material' ? sel.id : sel?.kind === 'object' ? exp.objects.find((o) => o.id === sel.id)?.material_id ?? null : null
    if (!matId) { toast('Select an object or material first', 'error'); return }
    const m = exp.materials.find((x) => x.id === matId)
    if (m?.dyes.some((d) => d.fluorophore === name)) { toast(`${name} is already in ${m.name}`); return }
    edit((e) => { e.materials.find((x) => x.id === matId)?.dyes.push({ fluorophore: name, concentration_uM: 10 }) })
    toast(`${name} added to ${m?.name}`)
  }

  return (
    <>
      <div className="ptitle">Library</div>
      <div className="libtabs"><Seg small value={tab} onChange={setTab} options={[{ value: 'ware', label: 'Lab-ware' }, { value: 'materials', label: 'Materials' }, { value: 'dyes', label: 'Dyes' }, { value: 'hardware', label: 'Hardware' }]} /></div>
      <div className="scroll">
        {tab === 'ware' && <div className="lib">
          {LIBRARY_SHAPES.map((s) => (
            <button key={s.key} draggable onDragStart={(e) => { e.dataTransfer.setData('application/x-sf-shape', s.key); e.dataTransfer.effectAllowed = 'copy' }} onClick={() => addObject(s.key)}>
              <ShapeIcon kind={KIND[s.key]} size={18} />{s.label}<small>{s.hint}</small>
            </button>
          ))}
        </div>}
        {tab === 'materials' && <div className="tree" style={{ paddingBottom: 10 }}>
          <div className="hint" style={{ padding: '0 8px 6px' }}>{sel?.kind === 'object' ? 'Click to assign to the selected object.' : 'Click to add to the experiment.'}</div>
          {BUNDLE.materials.map((m) => (
            <div key={m.id} className="node" onClick={() => assignMaterial(m.id)}>
              <span className="swatch" style={{ background: m.display_color ?? '#888' }} /><span className="nm">{m.name}</span>
              <span className="acts"><Plus size={13} /></span>
            </div>
          ))}
        </div>}
        {tab === 'dyes' && <div className="tree" style={{ paddingBottom: 10 }}>
          <div className="hint" style={{ padding: '0 8px 6px' }}>Click to add a dye (10 µM) to the selected object's material.</div>
          {BUNDLE.fluorophores.map((f) => (
            <Tip key={f.name} title={f.name} text={`ex ${f.ex_peak_nm} / em ${f.em_peak_nm} nm · Φ ${f.quantum_yield}`}>
              <div className="node" onClick={() => addDye(f.name)}>
                <span className="swatch" style={{ background: f.display_color ?? wavelengthToHex(f.em_peak_nm) }} /><span className="nm">{f.name}</span>
                <span className="tag">{f.ex_peak_nm}→{f.em_peak_nm}</span>
              </div>
            </Tip>
          ))}
        </div>}
        {tab === 'hardware' && <div className="lib">
          <button onClick={() => addSource('tunable_lamp')}><SourceIcon size={18} />Tunable lamp<small>Xe + monochromator</small></button>
          <button onClick={() => addSource('led_array')}><SourceIcon kind="led_array" size={18} />LED array<small>asymmetric line</small></button>
          <button onClick={() => addSource('laser')}><SourceIcon kind="laser" size={18} />Laser<small>narrow line</small></button>
          <button onClick={() => addCamera()}><CameraIcon size={18} />HSI camera<small>copy of the first</small></button>
        </div>}
      </div>
    </>
  )
}

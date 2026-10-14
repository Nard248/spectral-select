import { useEffect, useState } from 'react'
import { Plus } from 'lucide-react'
import { useStudio } from '../state/store'
import { LIBRARY_SHAPES, BLOCK_ITEMS } from '../state/library'
import { BUNDLE, clone } from '../state/presets'
import { libraryMaterial, tissueLibrary } from '../state/tissueLibrary'
import { ShapeIcon, SourceIcon, CameraIcon } from '../ui/icons'
import { Seg, Tip } from '../ui/controls'
import { wavelengthToHex } from '../lib/color'
import { toast } from '../ui/toast'
import { useResearch } from '../research/store'
import { loadCalibrated } from '../research/actions'
import type { CalibratedFluorophore } from '../api/researchTypes'
import type { SceneObject } from '../generated/experiment'

const KIND: Record<string, SceneObject['shape']['kind']> = { petri: 'petri_dish', cuvette: 'cuvette', slide: 'box', droplet: 'droplet', bead: 'sphere', block: 'box', puck: 'cylinder', block_wedge: 'layered_block', block_tissue: 'layered_block', block_vessel: 'layered_block', block_tumour: 'layered_block', block_skin: 'layered_block', block_two: 'layered_block' }

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
    const preset = libraryMaterial(id)!
    const existing = exp.materials.find((m) => m.id === id)
    if (!existing) edit((e) => { e.materials.push(clone(preset)) })
    if (sel?.kind === 'object') {
      edit((e) => { const o = e.objects.find((x) => x.id === sel.id); if (o) o.material_id = matId })
      toast(`${preset.name} → ${exp.objects.find((o) => o.id === sel.id)?.name}`)
    } else { select({ kind: 'material', id: matId }); toast(`${preset.name} added to experiment`) }
  }
  const calibrated = useResearch((r) => r.calibrated)
  useEffect(() => { if (tab === 'dyes') void loadCalibrated() }, [tab])
  const addCalibrated = (c: CalibratedFluorophore) => {
    if (!exp.fluorophores.some((f) => f.name === c.spec.name)) edit((e) => { e.fluorophores.push(clone(c.spec)) })
    addDye(c.spec.name)
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
        {tab === 'ware' && <>
          <div className="lib">
            {LIBRARY_SHAPES.filter((s) => !BLOCK_ITEMS.includes(s)).map((s) => (
              <button key={s.key} draggable onDragStart={(e) => { e.dataTransfer.setData('application/x-sf-shape', s.key); e.dataTransfer.effectAllowed = 'copy' }} onClick={() => addObject(s.key)}>
                <ShapeIcon kind={KIND[s.key]} size={18} />{s.label}<small>{s.hint}</small>
              </button>
            ))}
          </div>
          <div className="group-title" style={{ padding: '2px 14px 6px' }}>Layered blocks · 3D Monte Carlo</div>
          <div className="lib">
            {BLOCK_ITEMS.map((s) => (
              <button key={s.key} className="blk" draggable onDragStart={(e) => { e.dataTransfer.setData('application/x-sf-shape', s.key); e.dataTransfer.effectAllowed = 'copy' }} onClick={() => addObject(s.key)}>
                <ShapeIcon kind="layered_block" size={18} />{s.label}<small>{s.hint}</small>
              </button>
            ))}
          </div>
        </>}
        {tab === 'materials' && <div className="tree" style={{ paddingBottom: 10 }}>
          <div className="hint" style={{ padding: '0 8px 6px' }}>{sel?.kind === 'object' ? 'Click to assign to the selected object.' : 'Click to add to the experiment.'}</div>
          {BUNDLE.materials.filter((m) => !m.tissue).map((m) => (
            <div key={m.id} className="node" onClick={() => assignMaterial(m.id)}>
              <span className="swatch" style={{ background: m.display_color ?? '#888' }} /><span className="nm">{m.name}</span>
              <span className="acts"><Plus size={13} /></span>
            </div>
          ))}
          <div className="group-title" style={{ padding: '10px 8px 4px' }}>Tissue materials · chromophore optics</div>
          {tissueLibrary().map((m) => (
            <Tip key={m.id} title={m.name} text={`μs′(500) ${m.scattering.mus_prime_500_per_mm} /mm · blood ${((m.tissue?.blood_volume_fraction ?? 0) * 100).toFixed(1)} % · StO₂ ${((m.tissue?.oxygen_saturation ?? 0) * 100).toFixed(0)} % · water ${((m.tissue?.water_fraction ?? 0) * 100).toFixed(0)} %`}>
              <div className="node" onClick={() => assignMaterial(m.id)}>
                <span className="swatch" style={{ background: m.display_color ?? '#888' }} /><span className="nm">{m.name}</span>
                <span className="acts"><Plus size={13} /></span>
              </div>
            </Tip>
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
          {calibrated && calibrated.length > 0 && <>
            <div className="hint" style={{ padding: '10px 8px 4px', display: 'flex', alignItems: 'center', gap: 6, whiteSpace: 'nowrap' }}>
              <span className="tag" style={{ color: 'var(--ok)', background: 'rgba(78,242,138,.1)' }}>calibrated</span>fitted to real ME-HSI data
            </div>
            {calibrated.map((c) => {
              const p = c.provenance, f = c.spec
              return (
                <Tip key={f.name} title={f.name} text={`Fitted to ${p.dataset ?? 'real ME-HSI data'}${p.roi ? ` (${p.roi})` : ''} · ex ${f.ex_peak_nm.toFixed(0)} / em ${f.em_peak_nm.toFixed(0)} nm · R² ${p.r2 != null ? p.r2.toFixed(3) : 'n/a'} · SAM ${p.sam_mean_deg != null ? p.sam_mean_deg.toFixed(1) + '°' : 'n/a'} · rel. RMS ${p.rms_rel != null ? (p.rms_rel * 100).toFixed(1) + ' %' : 'n/a'}${p.notes ? ' · ' + p.notes : ''}`}>
                  <div className="node" onClick={() => addCalibrated(c)}>
                    <span className="swatch" style={{ background: f.display_color ?? wavelengthToHex(f.em_peak_nm) }} /><span className="nm">{f.name}</span>
                    <span className="tag" style={{ color: 'var(--ok)', background: 'rgba(78,242,138,.1)' }}>R² {p.r2 != null ? p.r2.toFixed(2) : '–'}</span>
                  </div>
                </Tip>
              )
            })}
          </>}
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

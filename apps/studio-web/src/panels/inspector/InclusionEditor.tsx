import { useMemo, useState } from 'react'
import { AlertTriangle, Circle, Copy, Pencil, Trash2 } from 'lucide-react'
import { useStudio } from '../../state/store'
import { useLayerView } from '../../state/layerView'
import { libraryMaterial } from '../../state/tissueLibrary'
import { clone } from '../../state/presets'
import { Num, Row, Sel, Seg, Section, Text, Tip } from '../../ui/controls'
import { NF, VF } from './Fields'
import {
  centerYForDepth, depthBelowTop, describeSize, INCLUSION_PRESETS, KINDS, makeInclusion, MAX_INCLUSIONS, topAt, uniqueName, validateInclusion,
  type Kind,
} from '../../lib/inclusions'
import type { Inclusion, LayeredBlockShape, MaterialSpec } from '../../generated/experiment'

const incs = (s: LayeredBlockShape) => s.inclusions as Inclusion[]

function InclusionCard({ objectId, shape, index, inc, materials, voxel, open, setOpen, upd }: {
  objectId: string; shape: LayeredBlockShape; index: number; inc: Inclusion; materials: MaterialSpec[]; voxel: number
  open: boolean; setOpen: (o: boolean) => void; upd: (fn: (s: LayeredBlockShape) => void, key?: string) => void
}) {
  const select = useStudio((s) => s.select)
  const setIncSel = useLayerView((s) => s.setIncSel)
  const mat = materials.find((m) => m.id === inc.material_id)
  const issues = validateInclusion(shape, inc, materials.map((m) => m.id), voxel)
  const bad = issues.some((i) => i.severity === 'error')
  const set = (fn: (i: Inclusion) => void, key?: string) => upd((s) => fn(incs(s)[index]), `inc${index}:${key ?? 'f'}`)
  const depth = depthBelowTop(shape, inc)
  const top = topAt(shape, inc.center_mm[0], inc.center_mm[2])
  const matOptions = materials.map((m) => ({ value: m.id, label: m.name }))
  return (
    <div className={`inc-card${open ? ' on' : ''}${bad ? ' bad' : ''}`} onMouseEnter={() => setIncSel(objectId, index)} onMouseLeave={() => setIncSel(objectId, null)}>
      <div className="inc-head">
        <span className="lyr-swatch" style={{ background: mat?.display_color ?? '#888', borderRadius: inc.kind === 'box' ? 3 : '50%' }} />
        <div className="lyr-name"><Text value={inc.name} onChange={(v) => set((i) => { i.name = v }, 'name')} placeholder="inclusion name" error={!inc.name.trim()} /></div>
        <Tip text="Duplicate"><button className="btn ghost icon sm" disabled={incs(shape).length >= MAX_INCLUSIONS} onClick={() => upd((s) => {
          const a = incs(s), c = clone(a[index]); c.name = uniqueName(c.name, a.map((x) => x.name)); c.center_mm = [c.center_mm[0] + 2, c.center_mm[1], c.center_mm[2] + 2]; a.splice(index + 1, 0, c)
        })}><Copy size={13} /></button></Tip>
        <Tip text="Delete"><button className="btn ghost icon sm danger" onClick={() => upd((s) => { incs(s).splice(index, 1) })}><Trash2 size={13} /></button></Tip>
      </div>
      <div className="inc-sub"><span>{describeSize(inc)}</span><span>depth {depth.toFixed(2)} mm</span><button className="btn ghost sm" style={{ marginLeft: 'auto', height: 18, padding: '0 6px', fontSize: 10.5 }} onClick={() => setOpen(!open)}>{open ? 'Collapse' : 'Edit'}</button></div>
      {open && <>
        <div className="lyr-mat">
          <Sel value={inc.material_id} options={matOptions.some((o) => o.value === inc.material_id) ? matOptions : [{ value: inc.material_id, label: `⚠ ${inc.material_id}` }, ...matOptions]} onChange={(v) => set((i) => { i.material_id = v }, 'mat')} />
          {mat && <Tip text="Edit this material"><button className="btn ghost icon sm" onClick={() => select({ kind: 'material', id: mat.id })}><Pencil size={13} /></button></Tip>}
        </div>
        <Seg small value={inc.kind} onChange={(k) => set((i) => { i.kind = k as Kind })} options={KINDS} />
        <VF label="Centre" tip="Centre (x, y, z) in the block's local frame: footprint centred on x = z = 0, bottom face at y = 0." value={inc.center_mm} step={0.1} unit="mm" onChange={(v) => set((i) => { i.center_mm = v }, 'c')} />
        <Row label="Depth" tip={`Distance of the centre below the top surface (top is at y = ${top.toFixed(2)} mm at this x, z). Negative = centre above the surface.`}>
          <Num value={+depth.toFixed(4)} min={-20} max={100} step={0.05} unit="mm" onChange={(d) => set((i) => { i.center_mm = [i.center_mm[0], centerYForDepth(shape, i.center_mm[0], i.center_mm[2], d), i.center_mm[2]] }, 'depth')} />
        </Row>
        {(inc.kind === 'sphere' || inc.kind === 'cylinder') && <NF label="Radius" value={inc.radius_mm} min={0.01} max={100} step={0.05} unit="mm" onChange={(v) => set((i) => { i.radius_mm = v }, 'r')} />}
        {inc.kind === 'ellipsoid' && <VF label="Semi-axes" tip="Semi-axes along x, y, z." value={inc.semi_axes_mm} step={0.05} unit="mm" onChange={(v) => set((i) => { i.semi_axes_mm = v.map((x) => Math.max(0.01, x)) as [number, number, number] }, 'sa')} />}
        {inc.kind === 'box' && <VF label="Size" tip="Full edge lengths along x, y, z." value={inc.size_mm} step={0.1} unit="mm" onChange={(v) => set((i) => { i.size_mm = v.map((x) => Math.max(0.01, x)) as [number, number, number] }, 'sz')} />}
        {inc.kind === 'cylinder' && <>
          <Row label="Axis" tip="Cylinder axis direction (a vessel running along x or z).">
            <Seg small value={inc.axis} onChange={(a) => set((i) => { i.axis = a })} options={[{ value: 'x', label: 'along x' }, { value: 'z', label: 'along z' }]} />
          </Row>
          <NF label="Length" tip="Cylinder length, clipped to the block. 1000 mm = through the whole block." value={inc.length_mm} min={0.05} max={5000} step={1} unit="mm" onChange={(v) => set((i) => { i.length_mm = v }, 'len')} />
        </>}
      </>}
      {issues.map((m, k) => m.severity === 'error'
        ? <div key={k} className="lyr-issue"><AlertTriangle size={13} />{m.message}</div>
        : <div key={k} className="hint" style={{ color: 'var(--warn)' }}>⚠ {m.message}</div>)}
    </div>
  )
}

/** Inclusion list of a `layered_block`: add (presets or custom) / duplicate / delete, kind, material, position (with a depth-below-top
 *  helper) and size per kind. Hovering a card highlights the inclusion in the 3D view. */
export function InclusionEditor({ objectId, shape }: { objectId: string; shape: LayeredBlockShape }) {
  const materials = useStudio((s) => s.exp.materials)
  const voxel = useStudio((s) => s.exp.acquisition.volume_mc.inclusion_voxel_mm)
  const edit = useStudio((s) => s.edit)
  const [openIdx, setOpenIdx] = useState<number | null>(null)
  const list = incs(shape)
  const upd = (fn: (s: LayeredBlockShape) => void, key?: string) => edit((e) => {
    const o = e.objects.find((x) => x.id === objectId); if (o && o.shape.kind === 'layered_block') fn(o.shape)
  }, key ? `obj:${objectId}:${key}` : undefined)
  const defaultMat = useMemo(() => materials.find((m) => m.tissue)?.id ?? shape.layers[shape.layers.length - 1]?.material_id ?? materials[0]?.id ?? 'water', [materials, shape.layers])
  const full = list.length >= MAX_INCLUSIONS
  const addPreset = (key: string) => {
    const p = INCLUSION_PRESETS.find((x) => x.key === key)!
    edit((e) => {   // one undo step: the library material (if missing) and the inclusion
      if (!e.materials.some((m) => m.id === p.materialId)) { const lm = libraryMaterial(p.materialId); if (lm) e.materials.push(clone(lm)) }
      const o = e.objects.find((x) => x.id === objectId); if (!o || o.shape.kind !== 'layered_block') return
      const a = o.shape.inclusions as Inclusion[]
      a.push(p.make(o.shape, e.materials.some((m) => m.id === p.materialId) ? p.materialId : defaultMat, uniqueName(p.label, a.map((x) => x.name))))
    })
    setOpenIdx(list.length)
  }
  const addCustom = (kind: Kind) => {
    upd((s) => { const a = incs(s); a.push(makeInclusion(kind, defaultMat, s, uniqueName('Inclusion', a.map((x) => x.name)))) })
    setOpenIdx(list.length)
  }
  return (
    <Section title="Inclusions" icon={<Circle size={13} />} extra={<span className="tag">{list.length}</span>} defaultOpen={list.length > 0}>
      <div className="inc-presets">
        {INCLUSION_PRESETS.map((p) => {
          const c = (libraryMaterial(p.materialId)?.display_color) ?? '#888'
          return (
            <Tip key={p.key} text={p.hint}>
              <button className="btn" disabled={full} onClick={() => addPreset(p.key)}><span className="sw" style={{ background: c }} />{p.label}<small>{p.hint.split(',')[0]}</small></button>
            </Tip>
          )
        })}
      </div>
      <div className="lyr-bar">
        <Sel value="" options={[{ value: '', label: '+ custom inclusion…' }, ...KINDS.map((k) => ({ value: k.value as string, label: k.label }))]} onChange={(v) => { if (v) addCustom(v as Kind) }} disabled={full} />
      </div>
      {list.length === 0 && <div className="hint">No inclusions. A sphere, ellipsoid, cylinder or box of another material replaces the layer material where it overlaps the block — a vessel, a tumour nodule, a fluorescent bead. Index-matched to the layer around it in the Monte Carlo; voxelised at {voxel} mm.</div>}
      <div className="lyr-list">
        {list.map((inc, i) => (
          <InclusionCard key={i} objectId={objectId} shape={shape} index={i} inc={inc} materials={materials} voxel={voxel} open={openIdx === i} setOpen={(o) => setOpenIdx(o ? i : null)} upd={upd} />
        ))}
      </div>
      {full && <div className="hint">At most {MAX_INCLUSIONS} inclusions per block.</div>}
    </Section>
  )
}

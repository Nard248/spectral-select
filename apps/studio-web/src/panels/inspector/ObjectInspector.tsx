import { useStudio } from '../../state/store'
import { materialMap, fluorophoreMap, BUNDLE, clone } from '../../state/presets'
import { LIBRARY_SHAPES, DEFAULT_BLOCK } from '../../state/library'
import { Section, Row, Sel, Toggle } from '../../ui/controls'
import { ShapeIcon } from '../../ui/icons'
import { Layers } from 'lucide-react'
import { InspectorHeader, HeaderActions, NF, TF, VF, SF, useErrAt } from './Fields'
import { wavelengthToHex } from '../../lib/color'
import type { SceneObject } from '../../generated/experiment'
import { LayerStackEditor } from './LayerStackEditor'
import { BlockViewSection } from './BlockViewSection'
import { totalRange } from '../../three/layers/profile'
import { uniformProfile } from '../../state/layerStack'

type Shape = SceneObject['shape']
const KIND_LABEL: Record<Shape['kind'], string> = { box: 'Box', sphere: 'Sphere', cylinder: 'Cylinder', droplet: 'Droplet', petri_dish: 'Petri dish', cuvette: 'Cuvette', layered_block: 'Layered block' }
const DEFAULT_SHAPE: Record<Shape['kind'], Shape> = {
  box: { kind: 'box', size_mm: [30, 10, 30] }, sphere: { kind: 'sphere', radius_mm: 7 }, cylinder: { kind: 'cylinder', radius_mm: 10, height_mm: 6 },
  droplet: { kind: 'droplet', base_radius_mm: 8, height_mm: 4 },
  petri_dish: { kind: 'petri_dish', inner_radius_mm: 17, wall_mm: 1, height_mm: 10, base_mm: 1, fill_height_mm: 4 },
  cuvette: { kind: 'cuvette', inner_mm: [10, 40, 10], wall_mm: 1.25, fill_height_mm: 30 },
  layered_block: { kind: 'layered_block', footprint_mm: [40, 40], layers: [{ name: 'substrate', material_id: 'tissue_substrate', thickness: uniformProfile(3) }, { name: 'cover', material_id: 'collagen_cover', thickness: uniformProfile(1) }] },
}

export function ObjectInspector({ id }: { id: string }) {
  const idx = useStudio((s) => s.exp.objects.findIndex((o) => o.id === id))
  const obj = useStudio((s) => s.exp.objects[idx])
  const materials = useStudio((s) => s.exp.materials)
  const edit = useStudio((s) => s.edit)
  const select = useStudio((s) => s.select)
  const dupe = useStudio((s) => s.duplicateSelected)
  const del = useStudio((s) => s.deleteSelected)
  const matErr = useErrAt(['objects', idx, 'material_id'])
  const vesErr = useErrAt(['objects', idx, 'vessel_material_id'])
  const exp = useStudio((s) => s.exp)
  if (!obj) return null
  const mats = materialMap({ materials }), fl = fluorophoreMap(exp)
  const upd = (fn: (o: SceneObject) => void, key?: string) => edit((e) => { const o = e.objects.find((x) => x.id === id); if (o) fn(o) }, key ?? `obj:${id}`)
  const sh = obj.shape
  const setShape = (patch: Record<string, unknown>) => upd((o) => { Object.assign(o.shape, patch) }, `obj:${id}:shape`)
  const m = mats[obj.material_id]
  const matOptions = materials.map((x) => ({ value: x.id, label: x.name }))
  const hasVessel = sh.kind === 'petri_dish' || sh.kind === 'cuvette'

  return (
    <>
      <InspectorHeader icon={<ShapeIcon kind={sh.kind} size={20} />} title={obj.name} sub={`${KIND_LABEL[sh.kind]} · ${obj.id}`}
        actions={<HeaderActions onDup={dupe} onDel={del} />} />
      <Section title="Object">
        <TF label="Name" value={obj.name} onChange={(v) => upd((o) => { o.name = v }, `obj:${id}:name`)} />
        <Row label="Visible"><Toggle value={obj.visible} onChange={(v) => upd((o) => { o.visible = v })} /></Row>
        <SF label="Shape" tip="Switching shape resets its parameters to defaults." value={sh.kind} options={(Object.keys(KIND_LABEL) as Shape['kind'][]).map((k) => ({ value: k, label: KIND_LABEL[k] }))}
          onChange={(k) => {
            if (k !== 'layered_block') { upd((o) => { o.shape = structuredClone(DEFAULT_SHAPE[k]) }); return }
            edit((e) => {   // bring the block's library materials along, then replace the shape (one undo step)
              for (const rid of DEFAULT_BLOCK.requires ?? []) { const bm = BUNDLE.materials.find((x) => x.id === rid); if (bm && !e.materials.some((x) => x.id === rid)) e.materials.push(clone(bm)) }
              const o = e.objects.find((x) => x.id === id); if (!o) return
              const blk = DEFAULT_BLOCK.make(id, o.material_id, null)
              o.shape = blk.shape; o.material_id = blk.material_id; o.vessel_material_id = null
              o.transform.position_mm[1] = 0
            })
          }} />
      </Section>
      <Section title="Transform">
        <VF label="Position" tip="Millimetres, Y-up, table top is y = 0." value={obj.transform.position_mm} step={1} unit="mm" onChange={(v) => upd((o) => { o.transform.position_mm = v }, `obj:${id}:pos`)} />
        <VF label="Rotation" tip="Euler XYZ in degrees." value={obj.transform.rotation_deg} step={1} unit="°" onChange={(v) => upd((o) => { o.transform.rotation_deg = v }, `obj:${id}:rot`)} />
      </Section>
      {sh.kind === 'layered_block' ? (
        <>
          <Section title="Layer stack" icon={<Layers size={13} />} extra={<span className="tag">{sh.layers.length} layers</span>}><LayerStackEditor objectId={id} shape={sh} /></Section>
          <BlockViewSection objectId={id} shape={sh} />
          <div style={{ padding: 12 }} className="hint">Each layer carries its own material (the object's single material is ignored). Light transport inside is solved by the 3D layered Monte Carlo — thickness {(() => { const r = totalRange(sh); return r.min === r.max ? `${+r.max.toFixed(3)} mm` : `${+r.min.toFixed(3)}–${+r.max.toFixed(3)} mm` })()}.</div>
        </>
      ) : (<>
      <Section title="Geometry">
        {sh.kind === 'box' && <VF label="Size" tip="Axis-aligned box centred on its origin (x, y, z)." value={sh.size_mm} step={0.5} unit="mm" onChange={(v) => setShape({ size_mm: v })} />}
        {sh.kind === 'sphere' && <NF label="Radius" value={sh.radius_mm} min={0.1} max={100} step={0.1} unit="mm" onChange={(v) => setShape({ radius_mm: v })} />}
        {sh.kind === 'cylinder' && <>
          <NF label="Radius" value={sh.radius_mm} min={0.1} max={100} step={0.1} unit="mm" onChange={(v) => setShape({ radius_mm: v })} />
          <NF label="Height" tip="Axis along local +Y, centred on the origin." value={sh.height_mm} min={0.1} max={100} step={0.1} unit="mm" onChange={(v) => setShape({ height_mm: v })} /></>}
        {sh.kind === 'droplet' && <>
          <NF label="Base radius" value={sh.base_radius_mm} min={0.1} max={60} step={0.1} unit="mm" onChange={(v) => setShape({ base_radius_mm: v })} />
          <NF label="Height" tip="Spherical cap; height ≤ base radius is a sessile drop (contact angle ≤ 90°)." value={sh.height_mm} min={0.1} max={60} step={0.1} unit="mm" onChange={(v) => setShape({ height_mm: v })} /></>}
        {sh.kind === 'petri_dish' && <>
          <NF label="Inner radius" value={sh.inner_radius_mm} min={1} max={100} step={0.5} unit="mm" onChange={(v) => setShape({ inner_radius_mm: v })} />
          <NF label="Wall" value={sh.wall_mm} min={0.1} max={10} step={0.1} unit="mm" onChange={(v) => setShape({ wall_mm: v })} />
          <NF label="Height" value={sh.height_mm} min={1} max={60} step={0.5} unit="mm" onChange={(v) => setShape({ height_mm: v })} />
          <NF label="Base" value={sh.base_mm} min={0.1} max={10} step={0.1} unit="mm" onChange={(v) => setShape({ base_mm: v })} />
          <NF label="Fill height" tip="Liquid depth above the inner base." value={sh.fill_height_mm} min={0} max={60} step={0.25} unit="mm" onChange={(v) => setShape({ fill_height_mm: v })} /></>}
        {sh.kind === 'cuvette' && <>
          <VF label="Inner size" tip="Inner cavity x, y, z." value={sh.inner_mm} step={0.5} unit="mm" onChange={(v) => setShape({ inner_mm: v })} />
          <NF label="Wall" value={sh.wall_mm} min={0.1} max={10} step={0.05} unit="mm" onChange={(v) => setShape({ wall_mm: v })} />
          <NF label="Fill height" value={sh.fill_height_mm} min={0} max={100} step={0.5} unit="mm" onChange={(v) => setShape({ fill_height_mm: v })} /></>}
      </Section>
      <Section title="Material">
        <Row label={hasVessel ? 'Contents' : 'Material'} error={matErr}>
          <Sel value={obj.material_id} options={matOptions.some((o) => o.value === obj.material_id) ? matOptions : [{ value: obj.material_id, label: `⚠ ${obj.material_id}` }, ...matOptions]}
            onChange={(v) => upd((o) => { o.material_id = v })} />
        </Row>
        {hasVessel && <Row label="Vessel" tip="Wall/base material of the dish or cuvette." error={vesErr}>
          <Sel value={obj.vessel_material_id ?? ''} options={[{ value: '', label: '— none —' }, ...matOptions]} onChange={(v) => upd((o) => { o.vessel_material_id = v || null })} />
        </Row>}
        {m && (
          <button className="btn" style={{ justifyContent: 'flex-start', height: 'auto', padding: '8px 10px' }} onClick={() => select({ kind: 'material', id: m.id })}>
            <span className="swatch" style={{ width: 14, height: 14, borderRadius: 4, background: m.display_color ?? '#888', flex: 'none' }} />
            <span style={{ textAlign: 'left', flex: 1, minWidth: 0 }}><div style={{ fontWeight: 600 }}>{m.name}</div>
              <div className="hint" style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>{m.dyes.length ? m.dyes.map((d) => <span key={d.fluorophore} style={{ color: wavelengthToHex(fl[d.fluorophore]?.em_peak_nm ?? 550) }}>{d.fluorophore} {d.concentration_uM} µM</span>) : 'no fluorophores'}</div></span>
            <span style={{ color: 'var(--accent)' }}>Edit →</span>
          </button>
        )}
      </Section>
      <div style={{ padding: 12 }} className="hint">Presets in library: {LIBRARY_SHAPES.length} lab-ware shapes. Drag the gizmo in the viewport or type exact values.</div>
      </>)}
    </>
  )
}

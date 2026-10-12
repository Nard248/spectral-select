import { useMemo, useState } from 'react'
import { Copy, GripVertical, Plus, Trash2, ArrowLeftRight, Pencil, AlertTriangle } from 'lucide-react'
import { useStudio } from '../../state/store'
import { uniqueId } from '../../lib/ids'
import { clone } from '../../state/presets'
import { addLayer, duplicateLayer, flipWedge, MAX_LAYERS, moveLayer, removeLayer, setProfileKind, setUniformThickness } from '../../state/layerStack'
import { interfaceHeights, profileEnds, totalRange, validateBlock, wedgeAngleDeg } from '../../three/layers/profile'
import { Num, Seg, Sel, Text, Tip } from '../../ui/controls'
import { toast } from '../../ui/toast'
import type { BlockLayer, LayeredBlockShape, MaterialSpec } from '../../generated/experiment'

const NEW = '__new__'
const DND = 'text/x-sf-layer'
const mmFmt = (t: number) => (t === 0 ? '0' : t < 0.1 ? `${(t * 1000).toFixed(0)} µm` : `${+t.toFixed(3)} mm`)

/** Live mini cross-section of the stack along one footprint axis (vertically exaggerated). */
export function StackDiagram({ shape, materials, axis, onAxis, hover }: { shape: LayeredBlockShape; materials: Record<string, MaterialSpec>; axis: 'x' | 'z'; onAxis: (a: 'x' | 'z') => void; hover: number | null }) {
  const W = 288, padX = 14, H = 104, base = 84
  const fp = shape.footprint_mm
  const size = axis === 'x' ? fp[0] : fp[1]
  const { max } = totalRange(shape)
  const sx = (W - 2 * padX) / Math.max(size, 1e-6)
  const sy = Math.min(60 / Math.max(max, 1e-6), sx * 14)
  const exag = sy / sx
  const N = 2   // interfaces are planes → two end samples describe them exactly
  const at = (u: number) => (axis === 'x' ? interfaceHeights(shape, u, 0) : interfaceHeights(shape, 0, u))
  const ends = [at(-size / 2), at(size / 2)]
  const X = (k: number) => padX + k * (W - 2 * padX) / (N - 1)
  const layers = shape.layers as BlockLayer[]
  return (
    <div className="lyr-diagram">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Layer stack cross-section">
        <defs>
          <linearGradient id="lyr-sheen" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#fff" stopOpacity=".16" /><stop offset="1" stopColor="#fff" stopOpacity="0" /></linearGradient>
        </defs>
        {layers.map((l, i) => {
          const c = materials[l.material_id]?.display_color ?? '#888'
          const pts = [[0, i], [1, i], [1, i + 1], [0, i + 1]].map(([k, j]) => `${X(k).toFixed(1)},${(base - ends[k][j] * sy).toFixed(1)}`).join(' ')
          const on = hover === i
          return <g key={i}>
            <polygon points={pts} fill={c} fillOpacity={on ? 0.95 : 0.78} stroke={on ? '#fff' : 'rgba(255,255,255,.55)'} strokeWidth={on ? 1.6 : 0.8} strokeLinejoin="round" />
            <polygon points={pts} fill="url(#lyr-sheen)" />
          </g>
        })}
        <line x1={padX} x2={W - padX} y1={base + 8} y2={base + 8} stroke="#9fb3c8" strokeWidth="1" />
        <line x1={padX} x2={padX} y1={base + 5} y2={base + 11} stroke="#9fb3c8" /><line x1={W - padX} x2={W - padX} y1={base + 5} y2={base + 11} stroke="#9fb3c8" />
        <text x={W / 2} y={base + 20} textAnchor="middle" fontSize="9.5" fill="#9fb3c8" fontFamily="var(--mono)">{size.toFixed(0)} mm along {axis}</text>
        <text x={W - padX} y={12} textAnchor="end" fontSize="9" fill="#5f6a7b" fontFamily="var(--mono)">vertical ×{exag < 1.05 ? '1' : exag.toFixed(1)}</text>
        <text x={padX} y={12} textAnchor="start" fontSize="9" fill="#5f6a7b" fontFamily="var(--mono)">−{axis}</text>
      </svg>
      <div className="lyr-axis"><Seg small value={axis} onChange={onAxis} options={[{ value: 'x', label: 'view along x' }, { value: 'z', label: 'view along z' }]} /></div>
    </div>
  )
}

function LayerCard({ objectId, shape, index, layer, material, materials, issues, drag, setDrag, hover, setHover, upd }: {
  objectId: string; shape: LayeredBlockShape; index: number; layer: BlockLayer; material?: MaterialSpec; materials: MaterialSpec[]
  issues: string[]; drag: { from: number; over: number | null } | null; setDrag: (d: { from: number; over: number | null } | null) => void
  hover: number | null; setHover: (i: number | null) => void; upd: (fn: (s: LayeredBlockShape) => void, key?: string) => void
}) {
  const select = useStudio((s) => s.select)
  const edit = useStudio((s) => s.edit)
  const nLayers = (shape.layers as BlockLayer[]).length
  const t = layer.thickness
  const setL = (fn: (l: BlockLayer) => void, key?: string) => upd((s) => fn((s.layers as BlockLayer[])[index]), key ?? `layer:${index}`)
  const matOptions = [...materials.map((m) => ({ value: m.id, label: m.name })), { value: NEW, label: `＋ New material from “${material?.name ?? 'material'}”…` }]
  const known = materials.some((m) => m.id === layer.material_id)
  const angle = wedgeAngleDeg(layer, shape.footprint_mm)
  const [a, b] = profileEnds(layer)
  const isDrop = drag && drag.over === index && drag.from !== index
  return (
    <div className={`lyr-card${hover === index ? ' hot' : ''}${issues.length ? ' bad' : ''}${drag?.from === index ? ' dragging' : ''}${isDrop ? (drag!.from < index ? ' drop-below' : ' drop-above') : ''}`}
      onMouseEnter={() => setHover(index)} onMouseLeave={() => setHover(null)}
      onDragOver={(e) => { if (e.dataTransfer.types.includes(DND)) { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; if (drag && drag.over !== index) setDrag({ ...drag, over: index }) } }}
      onDrop={(e) => {
        e.preventDefault()
        const from = Number(e.dataTransfer.getData(DND))   // the payload, not React state, is the source of truth
        if (Number.isInteger(from) && from !== index) upd((s) => { moveLayer(s, from, index) })
        setDrag(null)
      }}>
      <div className="lyr-head">
        <span className="lyr-grip" draggable title="Drag to reorder" onDragStart={(e) => { e.dataTransfer.setData(DND, String(index)); e.dataTransfer.effectAllowed = 'move'; setTimeout(() => setDrag({ from: index, over: null }), 0) }}   /* defer: restyling in dragstart makes Chrome cancel the drag */ onDragEnd={() => setDrag(null)}><GripVertical size={14} /></span>
        <span className="lyr-swatch" style={{ background: material?.display_color ?? '#888' }} />
        <div className="lyr-name"><Text value={layer.name} onChange={(v) => setL((l) => { l.name = v }, `layer:${index}:name`)} placeholder="layer name" error={issues.length > 0} /></div>
        <div className="lyr-acts">
          <Tip text="Duplicate layer"><button className="btn ghost icon sm" disabled={nLayers >= MAX_LAYERS} onClick={() => upd((s) => { duplicateLayer(s, index) })}><Copy size={13} /></button></Tip>
          <Tip text="Remove layer"><button className="btn ghost icon sm danger" disabled={nLayers <= 1} onClick={() => upd((s) => { removeLayer(s, index) })}><Trash2 size={13} /></button></Tip>
        </div>
      </div>
      <div className="lyr-mat">
        <Sel value={known ? layer.material_id : (layer.material_id as string)} options={known ? matOptions : [{ value: layer.material_id, label: `⚠ ${layer.material_id}` }, ...matOptions]}
          onChange={(v) => {
            if (v !== NEW) { setL((l) => { l.material_id = v }); return }
            const src = material ?? materials[0]; if (!src) return
            const id = uniqueId(`${src.id}_copy`, materials.map((m) => m.id))
            edit((e) => {   // one undo step: the copy and the re-assignment
              const c = clone(src); c.id = id; c.name = `${src.name} (${layer.name})`; e.materials.push(c)
              const o = e.objects.find((x) => x.id === objectId)
              if (o && o.shape.kind === 'layered_block') (o.shape.layers as BlockLayer[])[index].material_id = id
            })
            toast(`Created “${src.name} (${layer.name})” — fine-tune it in the material editor`)
          }} />
        {material && <Tip text="Edit this material"><button className="btn ghost icon sm" onClick={() => select({ kind: 'material', id: material.id })}><Pencil size={13} /></button></Tip>}
      </div>
      <div className="lyr-prof">
        <Seg small value={t.kind} onChange={(k) => setL((l) => setProfileKind(l, k))} options={[{ value: 'uniform', label: 'Uniform', title: 'Same thickness everywhere' }, { value: 'linear', label: 'Wedge', title: 'Thickness varies linearly across the block' }]} />
        {t.kind === 'linear' && <><span className="lyr-grow" /><span className="lyr-cap">along</span><Seg small value={t.axis} onChange={(ax) => setL((l) => { l.thickness.axis = ax })} options={[{ value: 'x', label: 'x', title: 'wedge along x' }, { value: 'z', label: 'z', title: 'wedge along z' }]} /></>}
      </div>
      <div className="lyr-prof">
        {t.kind === 'uniform'
          ? <div className="lyr-num"><Num value={t.thickness_mm} min={0} max={50} step={0.05} unit="mm" bar={false} onChange={(v) => setL((l) => setUniformThickness(l, v), `layer:${index}:t`)} /></div>
          : <>
            <div className="lyr-num"><Tip text={`Thickness at the −${t.axis} edge`}><Num value={t.thickness_mm} min={0} max={50} step={0.05} unit="mm" bar={false} onChange={(v) => setL((l) => { l.thickness.thickness_mm = v }, `layer:${index}:t0`)} /></Tip></div>
            <Tip text="Swap the two ends"><button className="btn ghost icon sm" onClick={() => setL((l) => flipWedge(l))}><ArrowLeftRight size={13} /></button></Tip>
            <div className="lyr-num"><Tip text={`Thickness at the +${t.axis} edge`}><Num value={t.end_thickness_mm} min={0} max={50} step={0.05} unit="mm" bar={false} onChange={(v) => setL((l) => { l.thickness.end_thickness_mm = v }, `layer:${index}:t1`)} /></Tip></div>
          </>}
      </div>
      {t.kind === 'linear' && <div className="lyr-note mono">{mmFmt(a)} → {mmFmt(b)} along {t.axis} · slope {angle?.toFixed(2)}°</div>}
      {issues.map((m, k) => <div key={k} className="err">{m}</div>)}
    </div>
  )
}

/** Inspector editor for a `layered_block`: footprint, ordered layer cards (drag to reorder), live cross-section. */
export function LayerStackEditor({ objectId, shape }: { objectId: string; shape: LayeredBlockShape }) {
  const materials = useStudio((s) => s.exp.materials)
  const edit = useStudio((s) => s.edit)
  const [axis, setAxis] = useState<'x' | 'z'>('x')
  const [hover, setHover] = useState<number | null>(null)
  const [drag, setDrag] = useState<{ from: number; over: number | null } | null>(null)
  const matMap = useMemo(() => Object.fromEntries(materials.map((m) => [m.id, m])), [materials])
  const layers = shape.layers as BlockLayer[]
  const issues = useMemo(() => validateBlock(shape, materials.map((m) => m.id)), [shape, materials])
  const globalIssues = issues.filter((i) => i.layer == null)
  const upd = (fn: (s: LayeredBlockShape) => void, key?: string) => edit((e) => {
    const o = e.objects.find((x) => x.id === objectId); if (o && o.shape.kind === 'layered_block') fn(o.shape)
  }, key ? `obj:${objectId}:${key}` : undefined)
  const idx = useStudio((s) => s.exp.objects.findIndex((o) => o.id === objectId))
  const serverMsgs = useStudio((s) => s.errors).filter((e) => {   // engine-side validation of this shape (path objects.<idx>.shape…)
    const loc = (e.loc[0] === 'body' ? e.loc.slice(1) : e.loc).map(String)
    return loc[0] === 'objects' && loc[1] === String(idx) && loc[2] === 'shape'
  }).map((e) => e.msg).filter((m) => !issues.some((i) => m.toLowerCase().includes(i.message.slice(0, 18).toLowerCase())))
  const range = totalRange(shape)
  const defaultMat = layers[layers.length - 1]?.material_id ?? materials[0]?.id ?? 'water'
  // show the TOP layer first (like a section drawing); data order stays bottom → top
  const order = layers.map((_, i) => layers.length - 1 - i)
  return (
    <>
      <div className="row" style={{ marginBottom: 2 }}>
        <label>Footprint</label>
        <div className="grid2">
          <Num value={shape.footprint_mm[0]} min={1} max={400} step={1} unit="x mm" bar={false} onChange={(v) => upd((s) => { s.footprint_mm = [v, s.footprint_mm[1]] }, 'fx')} />
          <Num value={shape.footprint_mm[1]} min={1} max={400} step={1} unit="z mm" bar={false} onChange={(v) => upd((s) => { s.footprint_mm = [s.footprint_mm[0], v] }, 'fz')} />
        </div>
      </div>
      <StackDiagram shape={shape} materials={matMap} axis={axis} onAxis={setAxis} hover={hover} />
      <div className="lyr-sum mono">{layers.length} layer{layers.length === 1 ? '' : 's'} · total {range.min === range.max ? mmFmt(range.max) : `${mmFmt(range.min)} – ${mmFmt(range.max)}`}</div>
      <div className="lyr-bar">
        <span className="lyr-cap">top</span>
        <button className="btn sm" disabled={layers.length >= MAX_LAYERS} onClick={() => upd((s) => { addLayer(s, defaultMat) })}><Plus size={13} />Add layer on top</button>
      </div>
      <div className="lyr-list">
        {order.map((i) => (
          <LayerCard key={`${i}`} objectId={objectId} shape={shape} index={i} layer={layers[i]} material={matMap[layers[i].material_id]} materials={materials}
            issues={issues.filter((x) => x.layer === i && x.severity === 'error').map((x) => x.message)} drag={drag} setDrag={setDrag} hover={hover} setHover={setHover} upd={upd} />
        ))}
      </div>
      <div className="lyr-cap" style={{ textAlign: 'center' }}>bottom (sits on the table)</div>
      {globalIssues.map((m, k) => <div key={k} className="lyr-issue"><AlertTriangle size={13} />{m.message}</div>)}
      {serverMsgs.map((m, k) => <div key={`s${k}`} className="lyr-issue"><AlertTriangle size={13} />{m}</div>)}
      {issues.filter((i) => i.layer != null && i.severity === 'warn').map((m, k) => <div key={`w${k}`} className="hint" style={{ color: 'var(--warn)' }}>⚠ {m.message}</div>)}
    </>
  )
}

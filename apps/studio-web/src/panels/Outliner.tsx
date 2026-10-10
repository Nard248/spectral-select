import { Eye, EyeOff, Trash2, Copy } from 'lucide-react'
import { useStudio, type SelKind } from '../state/store'
import { materialMap } from '../state/presets'
import { ShapeIcon, SourceIcon, CameraIcon, BackgroundIcon } from '../ui/icons'
import { Empty } from '../ui/controls'
import type { ReactNode } from 'react'

function Node({ kind, id, icon, name, swatch, hidden, onToggle, tag, removable = true }: { kind: SelKind; id: string; icon: ReactNode; name: string; swatch?: string | null; hidden?: boolean; onToggle?: () => void; tag?: string; removable?: boolean }) {
  const selected = useStudio((s) => s.selection?.kind === kind && s.selection.id === id)
  const select = useStudio((s) => s.select)
  const del = useStudio((s) => s.deleteSelected), dup = useStudio((s) => s.duplicateSelected)
  const setUi = useStudio((s) => s.setUi)
  return (
    <div className={`node${selected ? ' sel' : ''}${hidden ? ' hid' : ''}`} onClick={() => select({ kind, id })}
      onMouseEnter={() => setUi({ hoverId: id })} onMouseLeave={() => setUi({ hoverId: null })} role="button" tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter') select({ kind, id }) }}>
      <span className="ico">{icon}</span>
      <span className="nm">{name}</span>
      {tag && <span className="tag">{tag}</span>}
      {swatch && <span className="swatch" style={{ background: swatch }} />}
      <span className="acts">
        {onToggle && <button className="btn ghost icon sm" title={hidden ? 'Show' : 'Hide'} onClick={(e) => { e.stopPropagation(); onToggle() }}>{hidden ? <EyeOff size={13} /> : <Eye size={13} />}</button>}
        <button className="btn ghost icon sm" title="Duplicate (⌘D)" onClick={(e) => { e.stopPropagation(); select({ kind, id }); dup() }}><Copy size={13} /></button>
        {removable && <button className="btn ghost icon sm" title="Delete (Del)" onClick={(e) => { e.stopPropagation(); select({ kind, id }); del() }}><Trash2 size={13} /></button>}
      </span>
    </div>
  )
}

export function Outliner() {
  const exp = useStudio((s) => s.exp)
  const toggleVisible = useStudio((s) => s.toggleVisible)
  const mats = materialMap(exp)
  return (
    <>
      <div className="ptitle">Outliner</div>
      <div className="tree">
        <div className="group-title" style={{ margin: '2px 8px 4px' }}>Light sources</div>
        {exp.sources.map((s) => <Node key={s.id} kind="source" id={s.id} icon={<SourceIcon kind={s.kind} />} name={s.name} tag={s.id === exp.active_source_id ? 'active' : undefined} removable={exp.sources.length > 1} />)}
        <div className="group-title" style={{ margin: '8px 8px 4px' }}>Cameras</div>
        {exp.cameras.map((c) => <Node key={c.id} kind="camera" id={c.id} icon={<CameraIcon />} name={c.name} tag={c.id === exp.active_camera_id ? 'active' : undefined} removable={exp.cameras.length > 1} />)}
        <div className="group-title" style={{ margin: '8px 8px 4px' }}>Objects <span className="tag">{exp.objects.length}</span></div>
        {exp.objects.length === 0 && <Empty title="No samples yet">Click an item in the library below, or drag it into the viewport.</Empty>}
        {exp.objects.map((o) => <Node key={o.id} kind="object" id={o.id} icon={<ShapeIcon kind={o.shape.kind} />} name={o.name} swatch={mats[o.material_id]?.display_color} hidden={!o.visible} onToggle={() => toggleVisible(o.id)} />)}
        <div className="group-title" style={{ margin: '8px 8px 4px' }}>Environment</div>
        <Node kind="background" id="background" icon={<BackgroundIcon />} name={exp.background.kind === 'plane' ? `Table · ${mats[exp.background.material_id ?? '']?.name ?? 'none'}` : 'No table (void)'} swatch={mats[exp.background.material_id ?? '']?.display_color} removable={false} />
        <Node kind="acquisition" id="acquisition" icon={<ShapeIcon kind="box" />} name="Acquisition & physics" removable={false} />
      </div>
    </>
  )
}

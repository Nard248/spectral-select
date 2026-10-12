import { Eye, EyeOff, Trash2 } from 'lucide-react'
import { useAnalysis } from './store'
import { describe } from './roi'

export function RoiList() {
  const rois = useAnalysis((s) => s.rois)
  const selectedId = useAnalysis((s) => s.selectedId)
  const select = useAnalysis((s) => s.select)
  const update = useAnalysis((s) => s.updateRoi)
  const remove = useAnalysis((s) => s.removeRoi)
  if (!rois.length) return <div style={{ padding: '2px 12px 10px' }}><div className="hint">Draw regions with the tools above the image (<span className="kbd">D</span> disk, <span className="kbd">R</span> rectangle, <span className="kbd">G</span> polygon, <span className="kbd">L</span> line).</div></div>
  return (
    <div className="roi-list">
      {rois.map((r) => (
        <div key={r.id} className={`roi-item${r.id === selectedId ? ' sel' : ''}`} style={{ ['--c' as string]: r.color }} onClick={() => select(r.id)}>
          <span className="sw" style={{ opacity: r.visible ? 1 : 0.35 }} />
          <div className="nm">
            <input value={r.name} spellCheck={false} aria-label="ROI name" onClick={(e) => e.stopPropagation()} onChange={(e) => update(r.id, { name: e.target.value })}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === 'Escape') (e.target as HTMLInputElement).blur(); e.stopPropagation() }} />
            <small>{describe(r.spec)}</small>
          </div>
          <div className="ops">
            <button title={r.visible ? 'Hide' : 'Show'} aria-label={r.visible ? 'Hide ROI' : 'Show ROI'} onClick={(e) => { e.stopPropagation(); update(r.id, { visible: !r.visible }) }}>{r.visible ? <Eye size={13} /> : <EyeOff size={13} />}</button>
            <button className="del" title="Delete" aria-label="Delete ROI" onClick={(e) => { e.stopPropagation(); remove(r.id) }}><Trash2 size={13} /></button>
          </div>
        </div>))}
    </div>
  )
}

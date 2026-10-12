import { Circle, Crosshair, MousePointer2, Pentagon, RectangleHorizontal, Ruler, Trash2 } from 'lucide-react'
import { useAnalysis, type Tool } from './store'
import { Tip } from '../ui/controls'

const TOOLS: { tool: Tool; key: string; icon: React.ReactNode; title: string; tip: string }[] = [
  { tool: 'pointer', key: 'V', icon: <MousePointer2 size={14} />, title: 'Select / pick pixel', tip: 'Click a pixel to decompose it; click a region to select it; drag a region to move it or its handles to resize.' },
  { tool: 'point', key: 'P', icon: <Crosshair size={14} />, title: 'Point ROI', tip: 'A single-pixel ROI.' },
  { tool: 'disk', key: 'D', icon: <Circle size={14} />, title: 'Disk ROI', tip: 'Drag from the centre outward to set the radius.' },
  { tool: 'rect', key: 'R', icon: <RectangleHorizontal size={14} />, title: 'Rectangle ROI', tip: 'Drag from corner to corner.' },
  { tool: 'ellipse', key: 'E', icon: <Circle size={14} style={{ transform: 'scaleX(1.3)' }} />, title: 'Ellipse ROI', tip: 'Drag its bounding box.' },
  { tool: 'polygon', key: 'G', icon: <Pentagon size={14} />, title: 'Polygon ROI', tip: 'Click vertices; double-click, Enter or click the first vertex to close. Backspace removes the last vertex, Esc cancels.' },
  { tool: 'line', key: 'L', icon: <Ruler size={14} />, title: 'Line profile', tip: 'Drag a line — e.g. across the wedge — to read a profile of bands and ground truth along it.' },
]

export const TOOL_KEYS: Record<string, Tool> = Object.fromEntries(TOOLS.map((t) => [t.key.toLowerCase(), t.tool]))

export function RoiToolbar() {
  const tool = useAnalysis((s) => s.tool)
  const setTool = useAnalysis((s) => s.setTool)
  const n = useAnalysis((s) => s.rois.length)
  const clear = useAnalysis((s) => s.clearRois)
  return (
    <div className="roi-tools" role="toolbar" aria-label="ROI tools">
      {TOOLS.map((t) => (
        <Tip key={t.tool} title={`${t.title}  [${t.key}]`} text={t.tip}>
          <button className={tool === t.tool ? 'on' : ''} aria-label={t.title} aria-pressed={tool === t.tool} onClick={() => setTool(t.tool)}>{t.icon}<kbd>{t.key}</kbd></button>
        </Tip>
      )).flatMap((el, i) => (i === 1 ? [<span key="sep" className="vsep" />, el] : [el]))}
      <span className="vsep" />
      <Tip title="Delete all regions" text="Remove every ROI and line."><button disabled={!n} style={{ opacity: n ? 1 : 0.35 }} aria-label="Delete all ROIs" onClick={clear}><Trash2 size={14} /></button></Tip>
    </div>
  )
}

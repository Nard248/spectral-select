import type { ReactNode } from 'react'
import { Activity, Boxes, Grid3x3, Ruler, TrendingUp, Layers } from 'lucide-react'

/** Rounded dark plot well. */
export function ChartBox({ children, pad = true }: { children: ReactNode; pad?: boolean }) {
  return <div style={{ background: 'linear-gradient(180deg, #0b0e12, #0a0c10)', border: '1px solid var(--line)', borderRadius: 10, padding: pad ? '8px 6px 4px' : 0, overflow: 'visible' }}>{children}</div>
}

export function Busy({ on }: { on: boolean }) {
  return on ? <span className="hint" style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}><span className="spinner" style={{ width: 11, height: 11, margin: 0, borderWidth: 1.5 }} />updating</span> : null
}

const ICONS = { eem: <Grid3x3 size={20} />, line: <Ruler size={20} />, response: <TrendingUp size={20} />, light: <Boxes size={20} />, sweep: <Activity size={20} />, layers: <Layers size={20} /> }
export function WsEmpty({ icon, title, children }: { icon: keyof typeof ICONS; title: string; children?: ReactNode }) {
  return <div className="empty-ws"><div style={{ maxWidth: 340 }}><div className="big">{ICONS[icon]}</div><b>{title}</b><div style={{ lineHeight: 1.55 }}>{children}</div></div></div>
}

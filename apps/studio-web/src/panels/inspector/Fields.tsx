import type { ReactNode } from 'react'
import { Copy, Trash2 } from 'lucide-react'
import { useStudio } from '../../state/store'
import type { ValidationIssue } from '../../api/types'
import { Num, Row, Sel, Toggle, Text, Vec3 } from '../../ui/controls'

export const errAt = (errors: ValidationIssue[], path: (string | number)[]): string | undefined => {
  const key = path.join('.')
  return errors.find((e) => (e.loc[0] === 'body' ? e.loc.slice(1) : e.loc).join('.') === key)?.msg
}
export const useErrAt = (path: (string | number)[]) => useStudio((s) => errAt(s.errors, path))

interface NF {
  label: string; tip?: string; value: number; onChange: (v: number) => void
  min?: number; max?: number; step?: number; unit?: string; integer?: boolean; precision?: number; err?: string; bar?: boolean
}
export function NF({ label, tip, err, ...n }: NF) {
  return <Row label={label} tip={tip} error={err}><Num {...n} error={!!err} /></Row>
}
export function VF({ label, tip, value, onChange, step, unit }: { label: string; tip?: string; value: readonly number[]; onChange: (v: [number, number, number]) => void; step?: number; unit?: string }) {
  return <Row label={label} tip={tip}><Vec3 value={value} onChange={onChange} step={step} unit={unit} /></Row>
}
export function SF<T extends string | number>({ label, tip, value, options, onChange }: { label: string; tip?: string; value: T; options: { value: T; label: string }[] | readonly T[]; onChange: (v: T) => void }) {
  return <Row label={label} tip={tip}><Sel value={value} options={options} onChange={onChange} /></Row>
}
export function TF({ label, value, onChange, mono }: { label: string; value: string; onChange: (v: string) => void; mono?: boolean }) {
  return <Row label={label}><Text value={value} onChange={onChange} mono={mono} /></Row>
}
export function BF({ label, tip, value, onChange }: { label: string; tip?: string; value: boolean; onChange: (v: boolean) => void }) {
  return <Row label={label} tip={tip}><Toggle value={value} onChange={onChange} /></Row>
}
export function InspectorHeader({ icon, title, sub, actions }: { icon: ReactNode; title: string; sub?: string; actions?: ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '12px 12px 10px', borderBottom: '1px solid var(--line)', background: 'linear-gradient(180deg, rgba(53,216,255,.05), transparent)' }}>
      <span style={{ color: 'var(--accent)', display: 'inline-flex' }}>{icon}</span>
      <div style={{ minWidth: 0, flex: 1 }}><div style={{ fontWeight: 650, fontSize: 13.5, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{title}</div>{sub && <div className="hint">{sub}</div>}</div>
      {actions}
    </div>
  )
}

export function HeaderActions({ onDup, onDel, delDisabled, delTitle }: { onDup: () => void; onDel: () => void; delDisabled?: boolean; delTitle?: string }) {
  return (
    <div style={{ display: 'flex', gap: 4 }}>
      <button className="btn icon" title="Duplicate (⌘D)" onClick={onDup}><Copy size={14} /></button>
      <button className="btn icon danger" title={delTitle ?? 'Delete (Del)'} disabled={delDisabled} onClick={onDel}><Trash2 size={14} /></button>
    </div>
  )
}

import { useEffect, useRef, useState, type ReactNode } from 'react'
import { ChevronRight } from 'lucide-react'

/* ---------- tooltip ---------- */
export function Tip({ title, text, children, block }: { title?: string; text: string; children: ReactNode; block?: boolean }) {
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  return (
    <span style={{ display: block ? 'block' : 'contents' }}
      onMouseEnter={(e) => {
        const r = (e.currentTarget.firstElementChild ?? e.currentTarget).getBoundingClientRect()
        timer.current = setTimeout(() => setPos({ x: Math.min(r.left, window.innerWidth - 300), y: r.bottom + 6 }), 350)
      }}
      onMouseLeave={() => { if (timer.current) clearTimeout(timer.current); setPos(null) }}>
      {children}
      {pos && <div className="tt" style={{ left: pos.x, top: pos.y }}>{title && <b>{title}</b>}{text}</div>}
    </span>
  )
}

/* ---------- number field with drag-to-scrub ---------- */
interface NumProps {
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  unit?: string
  precision?: number
  disabled?: boolean
  error?: boolean
  integer?: boolean
  /** show a fill bar for finite ranges */
  bar?: boolean
}
const fmt = (v: number, precision: number) => (Number.isFinite(v) ? String(+v.toFixed(precision)) : '')

export function Num({ value, onChange, min = -Infinity, max = Infinity, step = 1, unit, precision, disabled, error, integer, bar = true }: NumProps) {
  const prec = precision ?? (integer ? 0 : step >= 1 ? 1 : Math.min(4, Math.ceil(-Math.log10(step)) + 1))
  const [text, setText] = useState<string | null>(null)
  const drag = useRef<{ x: number; v0: number; moved: boolean } | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const clamp = (v: number) => Math.min(max, Math.max(min, integer ? Math.round(v) : v))
  const finite = Number.isFinite(min) && Number.isFinite(max)
  const frac = finite ? Math.min(1, Math.max(0, (value - min) / (max - min || 1))) : 0
  const commit = (s: string) => {
    const v = parseFloat(s)
    if (Number.isFinite(v) && clamp(v) !== value) onChange(clamp(v))
    setText(null)
  }
  return (
    <div className={`num${error ? ' bad' : ''}`} style={disabled ? { opacity: 0.45, pointerEvents: 'none' } : undefined}
      onPointerDown={(e) => {
        if (text !== null || e.button !== 0) return
        drag.current = { x: e.clientX, v0: value, moved: false }
        ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
      }}
      onPointerMove={(e) => {
        const d = drag.current; if (!d) return
        const dx = e.clientX - d.x
        if (!d.moved && Math.abs(dx) < 3) return
        d.moved = true
        const sens = (finite && (max - min) < 2000 ? (max - min) / 260 : step) * (e.shiftKey ? 0.1 : 1) * (e.altKey ? 10 : 1)
        const raw = d.v0 + dx * sens
        const snapped = Math.round(raw / (step * (e.shiftKey ? 0.1 : 1))) * step * (e.shiftKey ? 0.1 : 1)
        const v = clamp(+snapped.toFixed(6))
        if (v !== value) onChange(v)
      }}
      onPointerUp={(e) => {
        const d = drag.current; drag.current = null
        ;(e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId)
        if (d && !d.moved) { inputRef.current?.focus(); inputRef.current?.select() }
      }}>
      {bar && finite && <div className="fill" style={{ width: `${frac * 100}%` }} />}
      <input ref={inputRef} className="mono" value={text ?? fmt(value, prec)} inputMode="decimal" spellCheck={false}
        onFocus={(e) => { setText(fmt(value, prec)); e.target.select() }}
        onChange={(e) => setText(e.target.value)}
        onBlur={(e) => commit(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
          else if (e.key === 'Escape') { setText(null); (e.target as HTMLInputElement).blur() }
          else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
            e.preventDefault()
            const mult = e.shiftKey ? 10 : e.altKey ? 0.1 : 1
            const nv = clamp(+(value + (e.key === 'ArrowUp' ? 1 : -1) * step * mult).toFixed(6))
            onChange(nv); setText(fmt(nv, prec))
          }
          e.stopPropagation()
        }} />
      {unit && <span className="unit">{unit}</span>}
    </div>
  )
}

export function Vec3({ value, onChange, step = 1, unit, min, max }: { value: readonly number[]; onChange: (v: [number, number, number]) => void; step?: number; unit?: string; min?: number; max?: number }) {
  return (
    <div className="grid3">
      {[0, 1, 2].map((i) => (
        <Num key={i} value={value[i]} step={step} min={min} max={max} bar={false} unit={i === 2 ? unit : undefined}
          onChange={(v) => { const n: [number, number, number] = [value[0], value[1], value[2]]; n[i] = v; onChange(n) }} />
      ))}
    </div>
  )
}

/* ---------- rows ---------- */
export function Row({ label, tip, children, stack, error }: { label?: ReactNode; tip?: string; children: ReactNode; stack?: boolean; error?: string }) {
  return (
    <div>
      <div className={`row${stack ? ' stack' : ''}`}>
        {label != null && (tip ? <Tip text={tip}><label>{label}</label></Tip> : <label>{label}</label>)}
        <div style={{ minWidth: 0 }}>{children}</div>
      </div>
      {error && <div className="err" style={{ marginLeft: stack ? 0 : 104 }}>{error}</div>}
    </div>
  )
}

export function Sel<T extends string | number>({ value, options, onChange, disabled }: { value: T; options: { value: T; label: string }[] | readonly T[]; onChange: (v: T) => void; disabled?: boolean }) {
  const opts = (options as readonly unknown[]).map((o) => (typeof o === 'object' ? (o as { value: T; label: string }) : { value: o as T, label: String(o) }))
  return (
    <select className="field" value={String(value)} disabled={disabled}
      onChange={(e) => { const o = opts.find((x) => String(x.value) === e.target.value); if (o) onChange(o.value) }}>
      {opts.map((o) => <option key={String(o.value)} value={String(o.value)}>{o.label}</option>)}
    </select>
  )
}

export function Toggle({ value, onChange, label, tip, title, disabled }: { value: boolean; onChange: (v: boolean) => void; label?: ReactNode; tip?: string; title?: string; disabled?: boolean }) {
  const el = (
    <span className={`tog${value ? ' on' : ''}`} role="switch" aria-checked={value} aria-disabled={disabled || undefined} tabIndex={disabled ? -1 : 0}
      style={disabled ? { opacity: 0.45, cursor: 'not-allowed' } : undefined}
      onClick={() => { if (!disabled) onChange(!value) }} onKeyDown={(e) => { if (!disabled && (e.key === ' ' || e.key === 'Enter')) { e.preventDefault(); onChange(!value) } }}>
      <span className="track" />{label != null && <span>{label}</span>}
    </span>
  )
  return tip ? <Tip title={title} text={tip}>{el}</Tip> : el
}

export function Text({ value, onChange, placeholder, mono, error }: { value: string; onChange: (v: string) => void; placeholder?: string; mono?: boolean; error?: boolean }) {
  const [t, setT] = useState<string | null>(null)
  return (
    <input className={`field${mono ? ' mono' : ''}`} style={error ? { borderColor: 'var(--err)' } : undefined} value={t ?? value} placeholder={placeholder} spellCheck={false}
      onChange={(e) => { setT(e.target.value); onChange(e.target.value) }} onBlur={() => setT(null)}
      onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); e.stopPropagation() }} />
  )
}

export function Seg<T extends string>({ value, options, onChange, small, accent }: { value: T; options: { value: T; label: ReactNode; title?: string }[]; onChange: (v: T) => void; small?: boolean; accent?: boolean }) {
  return (
    <div className={`seg${small ? ' small' : ''}${accent ? ' accent' : ''}`}>
      {options.map((o) => <button key={o.value} className={o.value === value ? 'on' : ''} title={o.title} onClick={() => onChange(o.value)}>{o.label}</button>)}
    </div>
  )
}

export function Section({ title, children, defaultOpen = true, extra, icon }: { title: ReactNode; children: ReactNode; defaultOpen?: boolean; extra?: ReactNode; icon?: ReactNode }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <section className={`sect${open ? ' open' : ''}`}>
      <header onClick={() => setOpen(!open)}>
        <span className="chev"><ChevronRight size={14} /></span>{icon}{title}
        {extra && <span className="extra" onClick={(e) => e.stopPropagation()}>{extra}</span>}
      </header>
      {open && <div className="body">{children}</div>}
    </section>
  )
}

export function GroupTitle({ children }: { children: ReactNode }) { return <div className="group-title">{children}</div> }

/* ---------- dropdown menu ---------- */
export function useOutside(ref: React.RefObject<HTMLElement | null>, cb: () => void, active: boolean) {
  useEffect(() => {
    if (!active) return
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) cb() }
    window.addEventListener('mousedown', h)
    return () => window.removeEventListener('mousedown', h)
  }, [active, cb, ref])
}

export function MenuButton({ label, icon, children, right, className = 'btn' }: { label: ReactNode; icon?: ReactNode; children: (close: () => void) => ReactNode; right?: boolean; className?: string }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const close = () => setOpen(false)
  useOutside(ref, close, open)
  return (
    <div className="menu-wrap" ref={ref}>
      <button className={className} onClick={() => setOpen(!open)}>{icon}{label}</button>
      {open && <div className={`menu${right ? ' right' : ''}`}>{children(close)}</div>}
    </div>
  )
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return <div className="empty"><b>{title}</b>{children}</div>
}

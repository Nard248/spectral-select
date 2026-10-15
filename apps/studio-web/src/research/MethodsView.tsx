import { Fragment, useEffect, useMemo } from 'react'
import { Check, ClipboardCopy, Download, FileText, Fingerprint, GitCommit, RefreshCw } from 'lucide-react'
import { useState } from 'react'
import { useStudio } from '../state/store'
import { Seg, Tip } from '../ui/controls'
import { downloadText, safeName } from '../analysis/exportData'
import { toast } from '../ui/toast'
import { useResearch } from './store'
import { parseMdBlocks } from './logic'
import { generateMethods, loadProvenance } from './actions'
import type { MethodsStyle, Provenance } from '../api/researchTypes'

/** Minimal Markdown for the Methods paragraph: **bold**, *italic*, `code`, blank-line paragraphs. */
export function MdParagraphs({ text }: { text: string }) {
  const blocks = useMemo(() => parseMdBlocks(text), [text])
  const inline = (s: string) => s.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g).map((t, i) =>
    t.startsWith('**') ? <b key={i}>{t.slice(2, -2)}</b> : t.startsWith('`') ? <code key={i}>{t.slice(1, -1)}</code> : t.startsWith('*') && t.length > 2 ? <i key={i}>{t.slice(1, -1)}</i> : <Fragment key={i}>{t}</Fragment>)
  return <>{blocks.map((b, i) => b.kind === 'h' ? <h4 key={i}>{inline(b.text)}</h4>
    : b.kind === 'p' ? <p key={i}>{inline(b.text)}</p>
      : <div key={i} className="rs-tablewrap"><table className="rs-mdtable"><thead><tr>{b.head.map((h, j) => <th key={j}>{inline(h)}</th>)}</tr></thead><tbody>{b.rows.map((r, j) => <tr key={j}>{r.map((c, k) => <td key={k}>{inline(c)}</td>)}</tr>)}</tbody></table></div>)}</>
}

async function copyText(t: string): Promise<boolean> {
  try { await navigator.clipboard.writeText(t); return true } catch {
    try { const ta = document.createElement('textarea'); ta.value = t; ta.style.position = 'fixed'; ta.style.opacity = '0'; document.body.appendChild(ta); ta.select(); const ok = document.execCommand('copy'); ta.remove(); return ok } catch { return false }
  }
}

const shortHash = (h: string | null | undefined, n = 12) => (h ? `${h.slice(0, n)}…` : '—')

function ProvenancePanel() {
  const { provenance, run } = useResearch()
  const prov = run?.manifest?.provenance ?? provenance
  const exHash = (run?.manifest?.provenance as { experiment_hash?: string } | undefined)?.experiment_hash
  const specHash = run?.manifest?.spec_hash
  const p = (provenance ?? prov) as Provenance | null
  const numba = typeof p?.numba === 'object' && p.numba ? `${p.numba.version}${p.numba.threading_layer ? ` · ${p.numba.threading_layer}` : ''}${p.numba.num_threads ? ` · ${p.numba.num_threads} threads` : ''}` : (p?.numba as string | null | undefined) ?? '—'
  const git = p?.git
  const [copied, setCopied] = useState<string | null>(null)
  const copy = async (k: string, v: string) => { if (await copyText(v)) { setCopied(k); setTimeout(() => setCopied(null), 1400) } }
  const rows: [string, React.ReactNode][] = p ? [
    ['Package', <><b>{p.package}</b> {p.package_version}</>],
    ['Engine', p.engine_version],
    ['Git commit', git?.commit ? <span className="mono"><GitCommit size={11} style={{ verticalAlign: -1 }} /> {git.commit.slice(0, 10)}{git.branch ? ` · ${git.branch}` : ''}{git.dirty ? <span className="rs-dirty" title="Uncommitted changes in the working tree when this ran"> dirty</span> : ''}</span> : '—'],
    ['Python', `${p.python} · numpy ${p.numpy} · scipy ${p.scipy}`],
    ['Numba', numba],
    ['Platform', `${p.platform} · ${p.machine} · ${p.cpu_count} CPU`],
    ['Recorded', p.timestamp_utc],
  ] : []
  return (
    <div className="lf-card">
      <header><Fingerprint size={14} style={{ color: 'var(--accent)' }} />Provenance <small>what produced these numbers</small><span className="sp" /><button className="btn sm ghost" onClick={() => void loadProvenance()}><RefreshCw size={12} />Refresh</button></header>
      <div className="cbody">
        {!p && <div className="hint">Provenance is read from the engine.</div>}
        {p && <dl className="rs-prov">{rows.map(([k, v]) => <Fragment key={k}><dt>{k}</dt><dd>{v}</dd></Fragment>)}</dl>}
        <div className="rs-hash">
          <div><span>Experiment hash</span><code title={exHash ?? ''}>{exHash ? shortHash(exHash, 24) : 'run a study to record it'}</code>{exHash && <button className="btn ghost sm icon" aria-label="Copy experiment hash" onClick={() => void copy('e', exHash)}>{copied === 'e' ? <Check size={12} /> : <ClipboardCopy size={12} />}</button>}</div>
          <div><span>Study spec hash</span><code title={specHash ?? ''}>{specHash ? shortHash(specHash, 24) : '—'}</code>{specHash && <button className="btn ghost sm icon" aria-label="Copy spec hash" onClick={() => void copy('s', specHash)}>{copied === 's' ? <Check size={12} /> : <ClipboardCopy size={12} />}</button>}</div>
        </div>
        <div className="hint" style={{ paddingTop: 6 }}>The experiment hash is the SHA-256 of the canonical JSON of the simulated experiment; with the commit, the seed and the library versions it makes a study exactly reproducible. Every export embeds this record.</div>
      </div>
    </div>
  )
}

export function MethodsView() {
  const { methodsStyle, methodsText, methodsFor, methodsBusy, methodsError, run, set } = useResearch()
  const exp = useStudio((s) => s.exp)
  const hasStudy = !!run
  const source = methodsFor === 'study' && hasStudy ? 'study' : 'experiment'
  useEffect(() => { void loadProvenance() }, [])
  useEffect(() => { void generateMethods(methodsStyle, source === 'study') /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [source, run?.id])
  const stem = safeName(`${source === 'study' ? run?.spec.name : exp.name}_methods`)
  const words = useMemo(() => methodsText.split(/\s+/).filter(Boolean).length, [methodsText])
  const [copied, setCopied] = useState(false)
  return (
    <div className="rs-methods">
      <div className="lf-card">
        <header><FileText size={14} style={{ color: 'var(--accent)' }} />Methods text <small>generated from the data model, so it cannot drift from what was run</small><span className="sp" />
          <Seg small value={source} onChange={(v) => set({ methodsFor: v })} options={[{ value: 'experiment', label: 'Experiment' }, { value: 'study', label: 'Experiment + study', title: hasStudy ? 'Adds the design, parameters, replicates, metrics and sensitivity estimators of the latest study' : 'Run a study first' }]} />
          <Seg small value={methodsStyle} onChange={(v: MethodsStyle) => { set({ methodsStyle: v }); void generateMethods(v, source === 'study') }} options={[{ value: 'markdown', label: 'Markdown' }, { value: 'latex', label: 'LaTeX' }]} /></header>
        <div className="cbody">
          {source === 'study' && !hasStudy && <div className="note warn" style={{ margin: '0 0 8px' }}>No study has been run yet; the text describes the current experiment and the study in the builder.</div>}
          {methodsError && <div className="note err" style={{ margin: '0 0 8px' }}>{methodsError}</div>}
          <div className={`rs-prose${methodsStyle === 'latex' ? ' tex' : ''}${methodsBusy ? ' busy' : ''}`} aria-live="polite">
            {!methodsText && methodsBusy && <span className="hint"><span className="spinner" style={{ width: 16, height: 16 }} />Writing…</span>}
            {methodsText && (methodsStyle === 'latex' ? <pre>{methodsText}</pre> : <MdParagraphs text={methodsText} />)}
          </div>
          <div className="rs-methods-foot">
            <span className="hint">{words} words · {methodsStyle === 'latex' ? 'paste into your .tex; add the reference to the bibliography' : 'paste into your manuscript'}</span><span className="sp" />
            <Tip text="Regenerate (e.g. after changing the experiment)"><button className="btn sm ghost" onClick={() => void generateMethods(methodsStyle, source === 'study')} disabled={methodsBusy}><RefreshCw size={12} />Regenerate</button></Tip>
            <button className="btn sm" disabled={!methodsText} onClick={() => downloadText(`${stem}.${methodsStyle === 'latex' ? 'tex' : 'md'}`, methodsText, 'text/plain')}><Download size={13} />Save</button>
            <button className="btn sm primary" disabled={!methodsText} onClick={async () => { const ok = await copyText(methodsText); setCopied(ok); if (ok) setTimeout(() => setCopied(false), 1500); else toast('Copy failed: select the text manually', 'error') }}>{copied ? <><Check size={13} />Copied</> : <><ClipboardCopy size={13} />Copy</>}</button>
          </div>
        </div>
      </div>
      <ProvenancePanel />
    </div>
  )
}

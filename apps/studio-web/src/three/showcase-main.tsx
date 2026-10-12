import { createRoot } from 'react-dom/client'
import { Showcase } from './Showcase'

const q = new URLSearchParams(location.search)
document.body.style.margin = '0'
createRoot(document.getElementById('root')!).render(
  <div style={{ width: '100vw', height: '100vh', background: '#06080c' }}>
    <Showcase view={q.get('view') ?? 'overview'} acquiring={q.get('acq') !== '0'} quality={q.get('quality') === 'low' ? 'low' : 'high'} selected={q.get('sel') ?? undefined} voidBackground={q.get('bg') === 'none'} stats={q.get('stats') === '1'} />
  </div>,
)

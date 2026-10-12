import { create } from 'zustand'

interface ToastState { msg: string | null; kind: 'info' | 'error'; show: (m: string, kind?: 'info' | 'error') => void }
let timer: ReturnType<typeof setTimeout> | null = null
export const useToast = create<ToastState>()((set) => ({
  msg: null, kind: 'info',
  show: (msg, kind = 'info') => {
    set({ msg, kind })
    if (timer) clearTimeout(timer)
    timer = setTimeout(() => set({ msg: null }), kind === 'error' ? 6000 : 3200)
  },
}))
export const toast = (m: string, kind: 'info' | 'error' = 'info') => useToast.getState().show(m, kind)

export function Toast() {
  const { msg, kind } = useToast()
  if (!msg) return null
  return <div className="toast" role="status" style={kind === 'error' ? { borderColor: 'var(--err)', color: '#ffc2c8' } : undefined}>{msg}</div>
}

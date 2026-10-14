// v1.2 Research workspace QA screenshots (1440x900) → screenshots/v12/.
// Usage: (vite on :5173 + engine on :8765)  node scripts/shoot-research.mjs [study|design|methods|inverse|builder|all] [url] [--mock]
import { chromium } from '@playwright/test'

const which = process.argv[2] ?? 'all'
const mock = process.argv.includes('--mock')
const base = (process.argv[3] && !process.argv[3].startsWith('--') ? process.argv[3] : 'http://localhost:5173/') + (mock ? '?engine=mock' : '')
const out = new URL('../screenshots/v12/', import.meta.url).pathname
const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 })
page.setDefaultTimeout(240000)
const logs = []
page.on('console', (m) => { if (['error', 'warning'].includes(m.type()) && !/GL Driver|Clock|403|GPU stall|swiftshader|WebGL/i.test(m.text())) logs.push(`[${m.type()}] ${m.text().slice(0, 300)}`) })
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`))

const open = async (preset = 'layered_wedge_phantom') => {
  await page.goto(base)
  await page.waitForFunction(() => window.studio && window.research)
  await page.waitForTimeout(1200)
  await page.evaluate((p) => { const s = window.studio.getState(); s.loadPreset(p); s.setUi({ showFrustum: false, showBeam: false, pip: false }) }, preset)
  await page.waitForTimeout(500)
}
const research = async (tab = 'study') => {
  await page.evaluate((t) => { window.studio.getState().setMode('research'); window.research.getState().set({ tab: t }) }, tab)
  await page.waitForTimeout(900)
}
const snap = async (name, ms = 700) => { await page.mouse.move(1000, 880); await page.waitForTimeout(ms); await page.screenshot({ path: out + name }); console.log('saved', name) }
const waitDone = () => page.waitForFunction(() => { const r = window.research.getState().run; return r && ['done', 'error', 'cancelled'].includes(r.state) }, null, { timeout: 600000 })
const scrollRes = (sel, off = 0) => page.evaluate(([s, o]) => { const sc = document.querySelector('.sw > .res'); const el = [...sc.querySelectorAll('.lf-card > header')].find((h) => h.textContent.includes(s)); if (el) sc.scrollTop += el.getBoundingClientRect().top - sc.getBoundingClientRect().top - 8 + o; return !!el }, [sel, off])

async function runStudy({ kind = 'saltelli', n = 16, reps = 2, mode = 'preview' } = {}) {
  await page.evaluate(([k, n, r, m]) => { const s = window.research.getState(); s.set({ design: { ...s.design, kind: k, n }, replicates: r, mode: m, touched: true }) }, [kind, n, reps, mode])
  await page.waitForTimeout(300)
  await page.evaluate(() => [...document.querySelectorAll('.runbar .btn.primary')].find((b) => /Run study/.test(b.textContent))?.click())
  await waitDone()
  await page.waitForTimeout(1500)
}

try {
  if (which === 'builder' || which === 'all') {
    await open(); await research('study'); await snap('r01-study-builder.png')
  }
  if (which === 'study' || which === 'all') {
    await open(); await research('study')
    await runStudy({ kind: 'saltelli', n: Number(process.env.N ?? 16), reps: Number(process.env.REPS ?? 2), mode: process.env.MODE ?? 'preview' })
    console.log(JSON.stringify(await page.evaluate(() => { const r = window.research.getState(); return { state: r.run?.state, n: r.rows.length, err: r.run?.error, sens: !!r.sens, sensErr: r.sensError } })))
    await snap('r02-study-results-sobol.png')
    await scrollRes('against', -4); await snap('r03-study-scatter.png')
    await scrollRes('Parallel', -4); await snap('r04-study-parallel.png')
    await scrollRes('Runs', -4); await snap('r05-study-runs.png')
  }
  if (which === 'design' || which === 'inverse' || which === 'all') {
    await open('tissue_under_collagen'); await research('inverse')
    await page.evaluate(() => window.research.getState().setInv({ tab: 'info' }))
    await page.waitForTimeout(500)
    await page.evaluate(() => [...document.querySelectorAll('.lf-card header .btn')].find((b) => /Compute/.test(b.textContent))?.click())
    await page.waitForFunction(() => ['done', 'error'].includes(window.research.getState().inv.fisher.state), null, { timeout: 600000 })
    console.log(JSON.stringify(await page.evaluate(() => { const f = window.research.getState().inv.fisher; return { s: f.state, e: f.error, n: f.result?.n_channels } })))
    await snap('r06-inverse-information.png')
    await page.evaluate(() => window.research.getState().setInv({ tab: 'design', k: 8 }))
    await page.waitForTimeout(400)
    await page.evaluate(() => [...document.querySelectorAll('.lf-card header .btn')].find((b) => /Select channels/.test(b.textContent))?.click())
    await page.waitForFunction(() => ['done', 'error'].includes(window.research.getState().inv.design.state), null, { timeout: 120000 })
    await snap('r07-optimal-design.png', 1200)
  }
  if (which === 'methods' || which === 'all') {
    await open(); await research('methods'); await page.waitForTimeout(1500); await snap('r08-methods-text.png')
  }
  if (which === 'calibrated' || which === 'all') {
    await open()
    await page.evaluate(() => window.studio.getState().select({ kind: 'object', id: window.studio.getState().exp.objects[0].id }))
    await page.getByRole('button', { name: /^Dyes$/ }).or(page.getByText(/^Dyes$/)).first().click()
    await page.waitForTimeout(2500)
    await page.locator('.node', { hasText: /collagen_cal/ }).first().hover()
    await snap('r09-calibrated-fluorophores.png', 900)
  }
} finally {
  console.log(logs.length ? logs.join('\n') : 'no console errors')
  await browser.close()
}

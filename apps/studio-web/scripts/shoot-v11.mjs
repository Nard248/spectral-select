// weblayers QA screenshots (1440x900) → screenshots/v11/.
// Usage: (vite on :5191, engine on :8765)  node scripts/shoot-v11.mjs [design|cutaway|stack|volume|lightfield|tour|all] [url]
// The light-field shots use the newest capture of the engine (run a layered-block capture first, e.g. via the UI or POST /api/captures).
import { chromium } from '@playwright/test'

const which = process.argv[2] ?? 'all'
const base = process.argv[3] ?? 'http://localhost:5191/'
const out = new URL('../screenshots/v11/', import.meta.url).pathname
const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 })
page.setDefaultTimeout(120000)
const logs = []
page.on('console', (m) => { if (['error', 'warning'].includes(m.type()) && !/GL Driver|Clock|403/.test(m.text())) logs.push(`[${m.type()}] ${m.text().slice(0, 300)}`) })
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`))

const open = async (preset = 'layered_wedge_phantom', ui = {}) => {
  await page.goto(base)
  await page.waitForFunction(() => window.studio && window.layerView)
  await page.waitForTimeout(1500)
  await page.evaluate(([p, ui]) => { const s = window.studio.getState(); s.loadPreset(p); s.setUi({ showFrustum: false, showBeam: false, pip: false, ...ui }) }, [preset, ui])
  await page.waitForTimeout(500)
}
const useLatestCapture = () => page.evaluate(() => {
  const caps = [...window.studio.getState().captures].sort((a, b) => b.created.localeCompare(a.created))
  window.studio.setState({ activeCaptureId: caps[0]?.id ?? null })
})
const fly = (pos, target) => page.evaluate(([p, t]) => window.__viewApi.flyTo(p, t, 10), [pos, target])
const snap = async (name, ms = 1800) => { await page.mouse.move(700, 640); await page.waitForTimeout(ms); await page.screenshot({ path: out + name }) }
const select = () => page.evaluate(() => window.studio.getState().select({ kind: 'object', id: 'block' }))
const inspectorTo = (title) => page.evaluate((t) => { const sc = document.querySelector('.panel.right .scroll'); const h = [...sc.querySelectorAll('header')].find((x) => x.textContent.includes(t)); h?.scrollIntoView({ block: 'start' }) }, title)

if (which === 'design' || which === 'all') {
  await open('layered_wedge_phantom', { showBeam: true })
  await useLatestCapture()
  await page.evaluate(() => window.studio.setState({ previewNm: 490 }))
  await select(); await fly([74, 52, 108], [0, 3, 0])
  await snap('01-block-design.png')
  await open('tissue_under_collagen')
  await select(); await fly([66, 46, 96], [0, 3, 0])
  await snap('01b-tissue-collagen-design.png')
}
if (which === 'cutaway' || which === 'all') {
  await open()
  await page.evaluate(() => window.studio.setState({ previewNm: 490 }))
  await select(); await page.waitForTimeout(400)
  await page.click('button[aria-label^="Cutaway"]')       // toggles + frames the section
  await snap('02-cutaway-wedge.png', 2600)
}
if (which === 'stack' || which === 'all') {
  await open(); await select(); await fly([66, 46, 100], [0, 3, 0])
  await page.waitForTimeout(500); await inspectorTo('Layer stack')
  await snap('03-layer-stack-inspector.png')
}
if (which === 'volume' || which === 'all') {
  await open(); await page.evaluate(() => window.studio.getState().select({ kind: 'acquisition', id: 'acquisition' }))
  await page.waitForTimeout(500); await inspectorTo('3D Monte Carlo')
  await snap('05-volume-mc-inspector.png', 800)
}
if (which === 'lightfield' || which === 'all') {
  await open()
  await useLatestCapture()
  await select()
  await page.evaluate(() => { window.layerView.getState().setLf({ on: true, kind: 'fluence', ex: 490, decades: 5 }) })
  await page.click('button[aria-label^="Cutaway"]')
  await snap('04-light-field.png', 4500)
  await page.evaluate(() => window.layerView.getState().setLf({ kind: 'source' }))
  await snap('04b-light-field-source.png', 3500)
  await page.evaluate(() => window.layerView.getState().setLf({ kind: 'absorbed' }))
  await snap('04c-light-field-absorbed.png', 3500)
}
if (which === 'tour' || which === 'all') {
  await open('mehsi_lab_replica')
  await useLatestCapture()
  await page.evaluate(() => window.sfTour.startTour({ step: 8 }))
  await snap('06-tour-layered.png', 4000)
  await page.evaluate(() => window.sfTour.goToStep(9)); await snap('06b-tour-cutaway.png', 5000)
  await page.evaluate(() => { window.layerView.getState().setLf({ ex: 490 }); window.sfTour.goToStep(11) }); await snap('06c-tour-lightfield.png', 6000)
  await page.evaluate(() => window.sfTour.endTour())
}
console.log(logs.slice(0, 20).join('\n') || 'no console errors')
await browser.close()

// v1.2 QA screenshots (1440x900) → screenshots/v12/.
// Usage: (vite running, e.g. `npx vite --port 5192`)  node scripts/shoot-v12.mjs [tissue|inclusions|cutaway|all] [url]
import { chromium } from '@playwright/test'

const which = process.argv[2] ?? 'all'
const base = process.argv[3] ?? 'http://localhost:5192/?engine=mock'
const out = new URL('../screenshots/v12/', import.meta.url).pathname
const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 })
page.setDefaultTimeout(120000)
const logs = []
page.on('console', (m) => { if (['error', 'warning'].includes(m.type()) && !/GL Driver|Clock|403|GPU stall/.test(m.text())) logs.push(`[${m.type()}] ${m.text().slice(0, 300)}`) })
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`))

const open = async (preset = 'generic_macro_bench') => {
  await page.goto(base)
  await page.waitForFunction(() => window.studio && window.layerView)
  await page.waitForTimeout(1500)
  await page.evaluate((p) => { const s = window.studio.getState(); s.loadPreset(p); s.setUi({ showFrustum: false, showBeam: false, pip: false }) }, preset)
  await page.waitForTimeout(400)
}
const fly = (pos, target) => page.evaluate(([p, t]) => window.__viewApi.flyTo(p, t, 10), [pos, target])
const snap = async (name, ms = 1500) => { await page.mouse.move(700, 640); await page.waitForTimeout(ms); await page.screenshot({ path: out + name }) }
// scroll the right inspector so the section whose title contains `t` sits at the top (opening it first)
const inspectorTo = (t, openIt = true) => page.evaluate(([t, o]) => {
  const sc = document.querySelector('.panel.right .scroll')
  const h = [...sc.querySelectorAll('section.sect > header')].find((x) => x.textContent.includes(t))
  if (!h) return false
  if (o && !h.parentElement.classList.contains('open')) h.click()
  sc.scrollTop += h.getBoundingClientRect().top - sc.getBoundingClientRect().top - 4
  return true
}, [t, openIt])

if (which === 'tissue' || which === 'all') {
  await open()
  await page.evaluate(() => { const s = window.studio.getState(); s.addObject('block_skin'); s.select({ kind: 'material', id: 'dermis_reticular' })
    s.edit((e) => { const m = e.materials.find((x) => x.id === 'dermis_reticular'); m.tissue.blood_volume_fraction = 0.03; m.tissue.melanosome_fraction = 0.01; m.tissue.beta_carotene_uM = 2 }) })
  await page.waitForTimeout(500)
  await inspectorTo('Tissue composition')
  await snap('01-tissue-composition.png', 800)
  await inspectorTo('Optical properties')
  await snap('02-tissue-optical-properties.png', 800)
}
if (which === 'inclusions' || which === 'all') {
  await open()
  const id = await page.evaluate(() => { const s = window.studio.getState(); const id = s.addObject('block_vessel'); s.addObject('block_tumour'); s.select({ kind: 'object', id }); return id })
  await page.evaluate((id) => { window.studio.getState().edit((e) => { const o = e.objects.find((x) => x.id === id); const p = o.shape.inclusions[0]; p.name = 'Vessel'
    o.shape.inclusions.push({ ...JSON.parse(JSON.stringify(p)), name: 'Tumour nodule', material_id: 'dysplasia_nodule', kind: 'sphere', radius_mm: 0.7, center_mm: [-6, 4.2, -5] }) }) }, id)
  await page.waitForTimeout(500)
  await page.evaluate(() => { const o = window.studio.getState().exp.objects; })
  await inspectorTo('Inclusions')
  // expand the first card
  await page.evaluate(() => { const b = [...document.querySelectorAll('.inc-card button')].find((x) => x.textContent === 'Edit'); b?.click() })
  await page.waitForTimeout(300)
  await snap('03-inclusion-editor.png', 800)
}
if (which === 'cutaway' || which === 'all') {
  await open()
  const id = await page.evaluate(() => { const s = window.studio.getState(); s.addObject('block_tumour'); const id = s.addObject('block_vessel')
    s.edit((e) => { e.objects = e.objects.filter((x) => x.id === id); const o = e.objects[0]; o.transform.position_mm = [0, 0, 0]
      o.shape.inclusions.push({ ...JSON.parse(JSON.stringify(o.shape.inclusions[0])), name: 'Tumour nodule', material_id: 'dysplasia_nodule', kind: 'sphere', radius_mm: 1.0, center_mm: [-7, 3.5, -0.3] }) })
    s.select({ kind: 'object', id }); return id })
  await page.waitForTimeout(500)
  await page.evaluate((id) => window.layerView.getState().setSection(id, { on: true, axis: 'z', position_mm: 0, keep: 'low', labels: true }), id)
  await page.waitForTimeout(500)
  await fly([6, 12, 34], [-1, 3, 0])
  await snap('04-vessel-under-collagen-cutaway.png', 9000)
  await fly([30, 20, 36], [0, 2.5, 0])
  await page.evaluate((id) => window.layerView.getState().setSection(id, { on: false }), id)
  await snap('05-vessel-under-collagen-design.png', 9000)
}
console.log(logs.slice(0, 20).join('\n') || 'no console errors')
await browser.close()

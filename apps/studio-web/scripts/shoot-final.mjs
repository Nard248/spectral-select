// v1.1 wrap-up QA screenshots (1440x900) → screenshots/v11/ with the REAL engine.
// Usage: (engine on :8765, `npm run dev` on :5173)  node scripts/shoot-final.mjs [wedge|pencil|all] [url]
//   wedge  : layered_wedge_phantom → Response panel with the layered-block defaults (ex near the dye peak, ex+40 nm band,
//            irradiance_normalized) + the Sweep panel
//   pencil : pencil_beam_penetration → light field XZ slice (z = 0), cutaway overlay in Design, line profile across the spot
import { chromium } from '@playwright/test'

const which = process.argv[2] ?? 'all'
const base = process.argv[3] ?? 'http://localhost:5173/'
const out = new URL('../screenshots/v11/', import.meta.url).pathname
const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 })
page.setDefaultTimeout(240000)
const logs = []
page.on('console', (m) => { if (['error', 'warning'].includes(m.type()) && !/GL Driver|Clock|403|GPU stall/.test(m.text())) logs.push(`[${m.type()}] ${m.text().slice(0, 300)}`) })
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`))

const open = async (preset, seq = null) => {
  await page.goto(base)
  await page.waitForFunction(() => window.studio && window.layerView && window.analysis && window.sweep)
  await page.waitForTimeout(2500)
  await page.evaluate(([p, seq]) => {
    const s = window.studio.getState(); s.loadPreset(p); s.setUi({ pip: false, showFrustum: false })
    if (seq) s.edit((e) => { e.sources[0].sequence = e.sources[0].sequence.filter((q) => seq.includes(q.wavelength_nm)) })
  }, [preset, seq])
  await page.waitForTimeout(600)
}
const capture = async () => {
  const before = await page.evaluate(() => window.studio.getState().captures.map((c) => c.id))
  await page.getByRole('button', { name: 'Acquire' }).click()
  await page.waitForTimeout(500)
  await page.getByRole('button', { name: /^Capture/ }).click()
  await page.waitForFunction((b) => { const s = window.studio.getState(); return s.captures.some((c) => !b.includes(c.id)) && s.activeCaptureId && !b.includes(s.activeCaptureId) }, before, { timeout: 900000 })
  await page.waitForTimeout(500)
}
const explore = async () => { await page.getByRole('button', { name: 'Explore' }).click(); await page.waitForTimeout(1500) }
const shot = async (name, ms = 1800) => { await page.waitForTimeout(ms); await page.screenshot({ path: out + name }) }

if (which === 'wedge' || which === 'all') {
  await open('layered_wedge_phantom', [450, 470, 490])
  await capture(); await explore()
  await page.evaluate(() => window.analysis.getState().setTab('response'))
  await page.waitForTimeout(2500)
  const st = await page.evaluate(() => ({ norm: window.analysis.getState().norm, metric: window.analysis.getState().response.metric }))
  console.log('defaults', JSON.stringify(st))
  const R = await page.evaluate(async () => {
    const a = window.analysis.getState(), id = window.studio.getState().activeCaptureId
    const r = await window.sfEngine().response(id, { x_ground_truth: 'layer_thickness:cover', metric: a.response.metric, roi: null, bins: 8, norm: { mode: a.norm.mode } })
    return { x: r.bin_centres.map((v) => +v.toFixed(2)), y: r.mean.map((v) => +v.toPrecision(4)) }
  })
  console.log('response', JSON.stringify(R))
  await shot('f01-wedge-response-defaults.png', 2500)
  // sweep panel defaults + a short preview sweep
  await page.evaluate(() => {
    const a = window.analysis.getState(); a.clearRois()
    a.addRoi({ shape: 'rect', x0: 60, y0: 40, x1: 110, y1: 80 }, 'ROI'); a.setMid('sweep')
  })
  await page.waitForTimeout(1000)
  console.log('sweep metric', JSON.stringify(await page.evaluate(() => window.sweep.getState().cfg.metric)))
  await page.evaluate(() => window.sweep.getState().setCfg({ steps: 5 }))
  await page.getByRole('button', { name: /Run sweep/ }).click()
  await page.waitForFunction(() => window.sweep.getState().run?.state === 'done' && window.sweep.getState().run.points.every((p) => p.metric != null || p.err), null, { timeout: 900000 })
  console.log('sweep points', JSON.stringify(await page.evaluate(() => window.sweep.getState().run.points.map((p) => [p.value, p.metric, p.err]))))
  await shot('f02-wedge-sweep-defaults.png', 1500)
}
if (which === 'pencil' || which === 'all') {
  await open('pencil_beam_penetration')
  await capture(); await explore()
  const spot = await page.evaluate(async () => {
    const eng = window.sfEngine(), id = window.studio.getState().activeCaptureId
    const g = await eng.groundTruth(id, 'irradiance@450')
    let best = -1, bi = 0
    for (let k = 0; k < g.data.length; k++) if (g.data[k] > best) { best = g.data[k]; bi = k }
    return { x: bi % g.width, y: Math.floor(bi / g.width), w: g.width, h: g.height }
  })
  console.log('spot', JSON.stringify(spot))
  await page.evaluate((s) => {
    const a = window.analysis.getState(); a.clearRois()
    a.addRoi({ shape: 'line', p0: [s.x - 40, s.y], p1: [s.x + 40, s.y], width_px: 3 }, 'across spot')
    a.setNorm({ mode: 'exposure_power' })   // a 1 mm beam has a vanishing irradiance tail; compare the raw spot with the beam profile
    a.setLine({ exNm: 450, bandsNm: [520], gt: 'irradiance@450' }); a.setTab('line')
  }, spot)
  await shot('f03-pencil-line-profile.png', 3000)
  await page.evaluate(() => window.analysis.getState().setMid('lightfield'))
  try { const d = page.locator('.cbar-h input').first(); await d.fill('5'); await d.press('Enter') } catch { /* keep the default range */ }
  await shot('f04-pencil-lightfield-xz.png', 4000)
  await page.evaluate(() => window.analysis.getState().setMid('image'))
  // cutaway overlay in Design (cut plane ⟂ z through the beam, z = 0)
  await page.getByRole('button', { name: 'Design' }).click()
  await page.waitForTimeout(1200)
  await page.evaluate(() => window.studio.getState().select({ kind: 'object', id: 'block' }))
  await page.waitForTimeout(500)
  await page.evaluate(() => window.layerView.getState().setLf({ on: true, kind: 'fluence', ex: 450, decades: 4 }))
  await page.click('button[aria-label^="Cutaway"]')
  await page.waitForTimeout(500)
  await page.evaluate(() => window.__viewApi.flyTo([34, 26, 48], [0, 3, 0], 10))
  await page.mouse.move(700, 640)
  await shot('f05-pencil-cutaway-overlay.png', 5000)
}
console.log(logs.slice(0, 25).join('\n') || 'no console errors')
await browser.close()

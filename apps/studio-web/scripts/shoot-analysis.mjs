// webanalysis QA screenshots (1440x900) → screenshots/v11/. Usage: node scripts/shoot-analysis.mjs [shot] [url]
// shots: eem | line | response | lightfield | sweep | draw | all     (needs the dev server on :5173 — the engine only accepts that origin; add --mock for the offline engine)
import { chromium } from '@playwright/test'

const which = process.argv[2] ?? 'all'
const base = process.argv[3] ?? 'http://localhost:5173/'
const mock = process.argv.includes('--mock')
const tag = mock ? 'wa-mock-' : 'wa-'   // wa- = webanalysis (weblayers owns the unprefixed v11 shots)
const out = new URL('../screenshots/v11/', import.meta.url).pathname
const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 })
page.setDefaultTimeout(180000)
const logs = []
page.on('console', (m) => { if (['error', 'warning'].includes(m.type())) logs.push(`[${m.type()}] ${m.text().slice(0, 300)}`) })
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`))

await page.goto(base + (mock ? '?engine=mock' : ''))
await page.waitForTimeout(2500)
await page.evaluate(() => {
  const s = window.studio.getState()
  s.loadPreset('layered_wedge_phantom')
  s.setUi({ pip: false })
  // real engine: a 3-excitation capture keeps the Monte Carlo under a minute
  s.edit((e) => { e.sources[0].sequence = e.sources[0].sequence.filter((q) => [380, 450, 490].includes(q.wavelength_nm)) })
})
await page.waitForTimeout(600)
const before = await page.evaluate(() => window.studio.getState().captures.map((c) => c.id))
await page.getByRole('button', { name: 'Acquire' }).click()
await page.waitForTimeout(500)
await page.getByRole('button', { name: /^Capture/ }).click()
await page.waitForFunction((b) => { const s = window.studio.getState(); return s.captures.some((c) => !b.includes(c.id)) && s.activeCaptureId && !b.includes(s.activeCaptureId) }, before, { timeout: 600000 })
await page.waitForTimeout(500)
await page.getByRole('button', { name: 'Explore' }).click()
await page.waitForTimeout(1500)

// locate the block through the cover-thickness ground truth
const geo = await page.evaluate(async () => {
  const eng = window.sfEngine()
  const id = window.studio.getState().activeCaptureId
  const gt = await eng.groundTruth(id, 'layer_thickness:cover')
  let x0 = 1e9, x1 = -1, y0 = 1e9, y1 = -1
  for (let k = 0; k < gt.data.length; k++) if (Number.isFinite(gt.data[k])) { const x = k % gt.width, y = Math.floor(k / gt.width); x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y) }
  const row = Math.round((y0 + y1) / 2)
  const left = gt.data[row * gt.width + x0 + 3], right = gt.data[row * gt.width + x1 - 3]
  return { x0, x1, y0, y1, w: gt.width, h: gt.height, thickLeft: left > right }
})
console.log('block', JSON.stringify(geo))
const fx = (f) => Math.round(geo.thickLeft ? geo.x0 + (geo.x1 - geo.x0) * f : geo.x1 - (geo.x1 - geo.x0) * f)   // f = 0 thick … 1 thin
const cy = Math.round((geo.y0 + geo.y1) / 2)

const addRois = () => page.evaluate(([a, b, y, r]) => {
  const st = window.analysis.getState()
  st.clearRois()
  st.addRoi({ shape: 'disk', x: a, y, radius_px: r }, 'thick cover')
  st.addRoi({ shape: 'disk', x: b, y, radius_px: r }, 'thin cover')
}, [fx(0.22), fx(0.8), cy, 7])
const shot = async (name) => { await page.waitForTimeout(1600); await page.screenshot({ path: out + tag + name }) }

if (which === 'draw') {
  // exercises the real pointer path: disk by drag, polygon by clicks, line by drag, move + resize
  const c = await page.locator('.imgstage canvas').first().boundingBox()
  const px = (x, y) => [c.x + ((x + 0.5) / geo.w) * c.width, c.y + ((y + 0.5) / geo.h) * c.height]
  await page.keyboard.press('d')
  let [ax, ay] = px(fx(0.2), cy); let [bx, by] = px(fx(0.2) + 8, cy)
  await page.mouse.move(ax, ay); await page.mouse.down(); await page.mouse.move(bx, by, { steps: 6 }); await page.mouse.up()
  await page.keyboard.press('g')
  for (const [x, y] of [[fx(0.5), cy - 12], [fx(0.7), cy - 12], [fx(0.7), cy + 12], [fx(0.5), cy + 12]]) { const [qx, qy] = px(x, y); await page.mouse.click(qx, qy) }
  await page.keyboard.press('Enter')
  await page.keyboard.press('l')
  ;[ax, ay] = px(geo.x0 + 3, geo.y1 - 6); [bx, by] = px(geo.x1 - 3, geo.y1 - 6)
  await page.mouse.move(ax, ay); await page.mouse.down(); await page.mouse.move(bx, by, { steps: 8 }); await page.mouse.up()
  const n = await page.evaluate(() => window.analysis.getState().rois.map((r) => r.spec.shape))
  console.log('drawn shapes', n.join(','))
  await shot('00-draw-tools.png')
}
if (which === 'eem' || which === 'all') {
  await addRois()
  await page.evaluate(() => { const s = window.analysis.getState(); s.setTab('eem'); s.setEem({ mode: 'overlay' }) })
  await shot('01-eem-two-rois.png')
  await page.evaluate(() => window.analysis.getState().setEem({ mode: 'diff' }))
  await shot('01b-eem-diff.png')
}
if (which === 'line' || which === 'all') {
  await addRois()
  await page.evaluate(([x0, x1, y]) => {
    const st = window.analysis.getState()
    st.addRoi({ shape: 'line', p0: [x0, y], p1: [x1, y], width_px: 5 }, 'across wedge')
    st.setTab('line')
  }, [geo.x0 + 3, geo.x1 - 3, cy])
  await shot('02-line-profile-wedge.png')
}
if (which === 'response' || which === 'all') {
  await addRois()
  await page.evaluate(() => window.analysis.getState().setTab('response'))
  await shot('03-response-vs-cover.png')
}
if (which === 'lightfield' || which === 'all') {
  await page.evaluate(() => window.analysis.getState().setMid('lightfield'))
  await shot('04-lightfield.png')
}
if (which === 'sweep' || which === 'all') {
  await addRois()
  await page.evaluate(() => { window.analysis.getState().setMid('sweep'); window.analysis.getState().setTab('response') })
  await page.waitForTimeout(800)
  await page.evaluate(() => window.sweep.getState().setCfg({ steps: 6 }))
  await page.getByRole('button', { name: /Run sweep/ }).click()
  await page.waitForFunction(() => window.sweep.getState().run?.state === 'done' && window.sweep.getState().run.points.every((p) => p.metric != null || p.err), null, { timeout: 900000 })
  await shot('05-sweep-results.png')
}
console.log(logs.slice(0, 25).join('\n') || 'no console errors')
await browser.close()

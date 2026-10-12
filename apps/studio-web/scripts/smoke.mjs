// UI smoke test: exercises most panels against the mock engine and fails on console errors.
import { chromium } from '@playwright/test'
const url = process.argv[2] ?? 'http://localhost:5173/'
const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } })
page.setDefaultTimeout(90000)
const errors = []
page.on('console', (m) => { if (m.type() === 'error' && !/500|ECONNREFUSED/.test(m.text())) errors.push(m.text()) })
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message))
await page.goto(url)
await page.waitForTimeout(2500)
await page.evaluate(() => window.studio.getState().setUi({ quality: 'low', pip: false }))
const S = (fn, arg) => page.evaluate(fn, arg)
let stepN = 0
const step = (n) => console.log(`step ${++stepN}: ${n}`)
const check = (cond, msg) => { if (!cond) { errors.push('CHECK FAILED: ' + msg) } }

// select every entity kind -> inspector renders
for (const sel of [{ kind: 'object', id: 'lichen_a' }, { kind: 'material', id: 'lichen_green' }, { kind: 'source', id: 'src_tls' }, { kind: 'camera', id: 'cam_nuance' }, { kind: 'background', id: 'background' }, { kind: 'acquisition', id: 'acquisition' }]) {
  await S((s) => window.studio.getState().select(s), sel)
  await page.waitForTimeout(250)
  const txt = await page.locator('.panel.right').innerText()
  check(txt.length > 80, `inspector empty for ${sel.kind}`)
}
step('library')
// library add + undo
const n0 = await S(() => window.studio.getState().exp.objects.length)
await page.getByRole('button', { name: /Petri dish/ }).click()
check((await S(() => window.studio.getState().exp.objects.length)) === n0 + 1, 'library add')
await page.keyboard.press('Control+z')
check((await S(() => window.studio.getState().exp.objects.length)) === n0, 'undo add')
step('typing')
// edit a number by typing
await S(() => window.studio.getState().select({ kind: 'object', id: 'lichen_a' }))
await page.waitForTimeout(200)
const posInput = page.locator('.panel.right .num input').first()
await posInput.click(); await posInput.fill('-31'); await posInput.press('Enter')
check((await S(() => window.studio.getState().exp.objects[0].transform.position_mm[0])) === -31, 'typed position')
step('physics')
// physics toggle
await S(() => window.studio.getState().select({ kind: 'acquisition', id: 'acquisition' }))
await page.waitForTimeout(200)
await page.locator('.toggles .tog').first().click()
check((await S(() => window.studio.getState().exp.acquisition.physics.fluorescence)) === false, 'physics toggle')
await page.locator('.toggles .tog').first().click()
step('capture')
// preview + capture
await page.keyboard.press('Space'); await page.waitForTimeout(1200)
check(!!(await S(() => window.studio.getState().previewMeta)), 'preview meta')
await page.getByRole('button', { name: /^Capture/ }).click()
await page.waitForFunction(() => window.studio.getState().captures.length > 0)
step('explore')
// explore: modes, roi, gt, compare
await page.getByRole('button', { name: 'Explore' }).click(); await page.waitForTimeout(800)
await page.getByRole('button', { name: 'True colour' }).click(); await page.waitForTimeout(500)
await page.getByRole('button', { name: 'False colour' }).click(); await page.waitForTimeout(500)
await page.getByRole('button', { name: 'Band', exact: true }).click()
const canvas = page.locator('.imgstage canvas').first(); const box = await canvas.boundingBox()
await page.mouse.click(box.x + box.width * 0.3, box.y + box.height * 0.4); await page.waitForTimeout(500)
await page.locator('.explore button[aria-label="Rectangle ROI"]').click()
await page.mouse.move(box.x + 40, box.y + 40); await page.mouse.down(); await page.mouse.move(box.x + 140, box.y + 120); await page.mouse.up()
await page.waitForTimeout(500)
check((await page.locator('.roi-item').count()) === 1, 'roi listed')
check((await page.locator('.explore .col').last().innerText()).includes('EEM'), 'eem panel opens for the new ROI')
await page.locator('.explore select.field').nth(1).selectOption({ index: 1 }); await page.waitForTimeout(600) // gt overlay
await page.locator('.explore select.field').nth(0).selectOption({ index: 1 }); await page.waitForTimeout(1200) // compare (preview)
check((await page.locator('.imgstage canvas').count()) === 2, 'compare shows two canvases')
await page.keyboard.press('ArrowRight'); await page.keyboard.press('ArrowDown')
step('back')
await page.getByRole('button', { name: 'Design' }).click(); await page.waitForTimeout(500)
console.log(errors.length ? errors.join('\n') : 'SMOKE OK')
await browser.close()
process.exit(errors.length ? 1 : 0)

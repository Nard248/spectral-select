// Takes the three reference screenshots (Design, Acquire mid-capture, Explore with a pixel) at 1440x900.
// Usage: npm run dev (other terminal), then: node scripts/screenshots.mjs [url]
import { chromium } from '@playwright/test'

const url = process.argv[2] ?? 'http://localhost:5173/'
const out = new URL('../screenshots/', import.meta.url).pathname
const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 })
page.setDefaultTimeout(120000)
const logs = []
page.on('console', (m) => { if (['error', 'warning'].includes(m.type())) logs.push(`[${m.type()}] ${m.text()}`) })
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`))
await page.goto(url)
await page.waitForTimeout(3500)
await page.screenshot({ path: out + '01-design.png' })

// Design with a selected object + inspector
await page.evaluate(() => window.studio.getState().select({ kind: 'object', id: 'lichen_a' }))
await page.waitForTimeout(1200)
await page.screenshot({ path: out + '02-design-selected.png' })

// Acquire: start a capture and grab it mid-way
await page.getByRole('button', { name: 'Acquire' }).click()
await page.waitForTimeout(600)
await page.getByRole('button', { name: /^Capture/ }).click()
await page.waitForTimeout(1800)
await page.screenshot({ path: out + '03-acquire.png' })
await page.waitForFunction(() => window.studio.getState().captures.length > 0, null, { timeout: 30000 })
await page.waitForTimeout(500)

// Explore: pick a bright pixel on a sample, at a fluorescing excitation
await page.getByRole('button', { name: 'Explore' }).click()
await page.waitForTimeout(1200)
await page.locator('.tab', { hasText: '415' }).first().click()
await page.waitForTimeout(1200)
await page.locator('input[type=range]').first().fill('12')
await page.waitForTimeout(600)
const canvas = page.locator('.imgstage canvas').first()
const box = await canvas.boundingBox()
const hit = await canvas.evaluate((c) => {
  const g = c.getContext('2d'); const d = g.getImageData(0, 0, c.width, c.height).data
  let best = 0, bi = 0
  for (let i = 0; i < d.length; i += 4) { const v = d[i] + d[i + 1] + d[i + 2]; if (v > best) { best = v; bi = i / 4 } }
  return { x: bi % c.width, y: Math.floor(bi / c.width), w: c.width, h: c.height }
})
if (box) await page.mouse.click(box.x + ((hit.x + 0.5) / hit.w) * box.width, box.y + ((hit.y + 0.5) / hit.h) * box.height)
await page.waitForTimeout(1200)
await page.screenshot({ path: out + '04-explore.png' })
console.log(logs.slice(0, 20).join('\n') || 'no console errors')
await browser.close()

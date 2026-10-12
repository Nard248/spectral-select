// Walks the guided tour step by step and screenshots every step (1440x900) into screenshots/tour/.
// Usage: dev server (+ engine) running, then: node scripts/tour-shots.mjs [url] [WxH]
import { chromium } from '@playwright/test'
import fs from 'fs'

const url = process.argv[2] ?? 'http://localhost:5173/'
const [W, H] = (process.argv[3] ?? '1440x900').split('x').map(Number)
const out = new URL(W === 1440 ? '../screenshots/tour/' : `../screenshots/tour-${W}x${H}/`, import.meta.url).pathname
fs.mkdirSync(out, { recursive: true })
const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width: W, height: H } })
page.setDefaultTimeout(180000)
const logs = []
page.on('console', (m) => { if (m.type() === 'error') logs.push(m.text()) })
page.on('pageerror', (e) => logs.push('PAGEERR ' + e.message))
await page.goto(url)
await page.waitForTimeout(3500)
const total = await page.evaluate(() => { window.sfTour.startTour({ step: 0 }); return 8 })
for (let i = 0; i < total; i++) {
  if (i > 0) await page.evaluate((i) => window.sfTour.goToStep(i), i)
  await page.waitForFunction(() => !window.__tourBusy && true, null, { timeout: 1000 }).catch(() => {})
  // wait until the step script is done (busy spinner gone)
  await page.waitForFunction(() => !document.querySelector('.tour-busy'), null, { timeout: 180000 })
  await page.waitForTimeout(1200)
  await page.screenshot({ path: `${out}step-${i + 1}.png` })
  console.log('step', i + 1, 'ok')
}
console.log(logs.slice(0, 10).join('\n') || 'no console errors')
await browser.close()

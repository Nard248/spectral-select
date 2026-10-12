// Tour smoke test (all steps incl. the layered-block ones) against the running engine / dev server.
// Usage: node scripts/tour-smoke.mjs [url]   → prints "TOUR SMOKE OK" or exits 1
import { chromium } from '@playwright/test'
const base = process.argv[2] ?? 'http://localhost:5173/?tour=1'
const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
page.setDefaultTimeout(120000)
const errs = []
page.on('console', (m) => { if (m.type() === 'error' && !/GL Driver|GPU stall|403|favicon/.test(m.text())) errs.push(m.text().slice(0, 200)) })
page.on('pageerror', (e) => errs.push(`pageerror ${e.message}`))
await page.goto(base)
await page.waitForSelector('[aria-label="Guided tour"]')
const total = await page.evaluate(() => Number(document.querySelector('.tour-step').textContent.split('/')[1]))
const titles = []
for (let i = 0; i < total; i++) {
  await page.evaluate((k) => window.sfTour.goToStep(k), i)
  await page.waitForTimeout(i === 4 || i === 5 || i === 10 ? 6000 : 2500)
  const t = await page.evaluate(() => ({ step: document.querySelector('.tour-step').textContent, text: document.querySelector('[aria-label="Guided tour"]').innerText.slice(0, 90).replace(/\n/g, ' ') }))
  titles.push(`${t.step}: ${t.text}`)
}
console.log(titles.join('\n'))
await page.screenshot({ path: new URL('../screenshots/v11/f06-tour-last-step.png', import.meta.url).pathname })
await page.evaluate(() => window.sfTour.endTour())
await page.waitForTimeout(500)
const closed = await page.locator('[aria-label="Guided tour"]').count()
await browser.close()
if (errs.length || closed) { console.log('ERRORS', closed ? 'tour still open' : '', errs.join('\n')); process.exit(1) }
console.log(`TOUR SMOKE OK (${total} steps)`)

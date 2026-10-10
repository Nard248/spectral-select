// Screenshots of showcase.html from several angles. Usage: node scripts/shoot-three.mjs [view ...] (dev server on :5199)
import { chromium } from '@playwright/test'
import { mkdirSync } from 'node:fs'

const views = process.argv.slice(2).length ? process.argv.slice(2) : ['overview', 'top', 'camera', 'lamp', 'led', 'laser', 'samples', 'side']
const out = new URL('../screenshots/three/', import.meta.url).pathname
mkdirSync(out, { recursive: true })
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--enable-webgl'] })
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
const errors = []
page.on('console', (m) => { if (m.text().startsWith('PERF')) console.log(m.text())
   if (['error', 'warning'].includes(m.type())) errors.push(`${m.type()}: ${m.text()}`) })
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message))
for (const v of views) {
  const extra = process.env.EXTRA ?? ''
  await page.goto(`http://localhost:5199/showcase.html?view=${v}${extra}`)
  await page.waitForTimeout(Number(process.env.WAIT ?? 9000))
  await page.screenshot({ path: `${out}${v}.png`, timeout: 180000 })
  console.log('saved', v)
}
console.log(errors.slice(0, 20).join('\n') || 'no console errors')
await browser.close()

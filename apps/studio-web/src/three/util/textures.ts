/** Procedural canvas textures (no network assets). All are lazily created singletons or keyed caches. */
import * as THREE from 'three'

const cache = new Map<string, THREE.Texture>()

function canvas(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement('canvas')
  c.width = w; c.height = h
  return [c, c.getContext('2d')!]
}

function finish(c: HTMLCanvasElement, srgb: boolean, repeat = false): THREE.CanvasTexture {
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace
  t.anisotropy = 8
  t.generateMipmaps = true
  t.minFilter = THREE.LinearMipmapLinearFilter
  if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping
  t.needsUpdate = true
  return t
}

function memo(key: string, make: () => THREE.Texture): THREE.Texture {
  let t = cache.get(key)
  if (!t) { t = make(); cache.set(key, t) }
  return t
}

/** Brushed-metal roughness map: fine streaks along u (horizontal). Tileable. */
export function brushedRoughness(): THREE.Texture {
  return memo('brushed', () => {
    const [c, g] = canvas(256, 256)
    g.fillStyle = '#8a8a8a'; g.fillRect(0, 0, 256, 256)
    let s = 12345
    const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296)
    for (let i = 0; i < 1400; i++) {
      const y = rnd() * 256, len = 30 + rnd() * 200, x = rnd() * 256
      const v = 90 + rnd() * 110
      g.strokeStyle = `rgba(${v},${v},${v},${0.15 + rnd() * 0.3})`
      g.lineWidth = 0.6 + rnd() * 0.8
      g.beginPath(); g.moveTo(x, y); g.lineTo(x + len, y + (rnd() - 0.5) * 0.6); g.stroke()
      if (x + len > 256) { g.beginPath(); g.moveTo(x - 256, y); g.lineTo(x - 256 + len, y); g.stroke() }
    }
    const t = finish(c, false, true); t.repeat.set(3, 3); return t
  })
}

/** Vertical knurl stripes (bump) for rotating rings: u wraps around the circumference. */
export function knurlBump(): THREE.Texture {
  return memo('knurl', () => {
    const [c, g] = canvas(256, 64)
    g.fillStyle = '#808080'; g.fillRect(0, 0, 256, 64)
    const n = 16
    for (let i = 0; i < n; i++) {
      const x = (i / n) * 256
      const gr = g.createLinearGradient(x, 0, x + 256 / n, 0)
      gr.addColorStop(0, '#202020'); gr.addColorStop(0.5, '#f0f0f0'); gr.addColorStop(1, '#202020')
      g.fillStyle = gr; g.fillRect(x, 0, 256 / n, 64)
    }
    const t = finish(c, false, true); t.repeat.set(4, 1); return t
  })
}

/** Scale ring (aperture / focus) with ticks + numerals, colour map. u wraps around. */
export function scaleRing(kind: 'aperture' | 'focus'): THREE.Texture {
  return memo('ring-' + kind, () => {
    const [c, g] = canvas(1024, 128)
    g.fillStyle = '#16181c'; g.fillRect(0, 0, 1024, 128)
    // knurl base
    for (let i = 0; i < 128; i++) { g.fillStyle = i % 2 ? '#1d2025' : '#111317'; g.fillRect(i * 8, 0, 4, 128) }
    g.fillStyle = '#e8ecf2'; g.strokeStyle = '#e8ecf2'
    g.font = 'bold 30px Helvetica, Arial, sans-serif'; g.textAlign = 'center'
    const marks = kind === 'aperture' ? ['1.4', '2', '2.8', '4', '5.6', '8', '11', '16', '22'] : ['0.3', '0.5', '1', '2', '5', '∞']
    marks.forEach((m, i) => {
      const x = 90 + (i * 800) / (marks.length - 1)
      g.fillText(m, x, 62)
      g.lineWidth = 3; g.beginPath(); g.moveTo(x, 78); g.lineTo(x, 112); g.stroke()
    })
    g.lineWidth = 1.5
    for (let i = 0; i <= 40; i++) { const x = 90 + (i * 800) / 40; g.beginPath(); g.moveTo(x, 100); g.lineTo(x, 112); g.stroke() }
    const t = finish(c, true, true); return t
  })
}

/** Rectangular engraved label plate: product name + small print. */
export function labelPlate(lines: string[], opts: { w?: number; h?: number; bg?: string; fg?: string; accent?: string } = {}): THREE.Texture {
  const key = 'label:' + lines.join('|') + JSON.stringify(opts)
  return memo(key, () => {
    const w = opts.w ?? 768, h = opts.h ?? 192
    const [c, g] = canvas(w, h)
    g.fillStyle = opts.bg ?? '#0e1014'; g.fillRect(0, 0, w, h)
    const gr = g.createLinearGradient(0, 0, w, h)
    gr.addColorStop(0, 'rgba(255,255,255,0.10)'); gr.addColorStop(0.5, 'rgba(255,255,255,0)'); gr.addColorStop(1, 'rgba(255,255,255,0.06)')
    g.fillStyle = gr; g.fillRect(0, 0, w, h)
    g.strokeStyle = opts.accent ?? '#5ac8fa'; g.lineWidth = 3; g.strokeRect(10, 10, w - 20, h - 20)
    g.fillStyle = opts.fg ?? '#eef3fa'; g.textAlign = 'left'; g.textBaseline = 'middle'
    // spectral gradient bar
    const bar = g.createLinearGradient(30, 0, 160, 0)
    ;['#7a2cff', '#2a5bff', '#19d3ff', '#3ff08a', '#ffe14d', '#ff8a2a', '#ff3a3a'].forEach((col, i, a) => bar.addColorStop(i / (a.length - 1), col))
    g.fillStyle = bar; g.fillRect(32, h * 0.2, 130, 14)
    g.fillStyle = opts.fg ?? '#eef3fa'
    lines.forEach((ln, i) => {
      const big = i === 0
      g.font = `${big ? 'bold 64' : '30'}px Helvetica, Arial, sans-serif`
      g.globalAlpha = big ? 1 : 0.72
      g.fillText(ln, 32, big ? h * 0.48 : h * 0.72 + (i - 1) * 34)
    })
    g.globalAlpha = 1
    return finish(c, true)
  })
}

/** Yellow warning triangle sticker (UV / laser). */
export function warningSticker(kind: 'uv' | 'laser'): THREE.Texture {
  return memo('warn-' + kind, () => {
    const [c, g] = canvas(256, 256)
    g.clearRect(0, 0, 256, 256)
    g.fillStyle = '#ffd200'; g.strokeStyle = '#111'; g.lineWidth = 12; g.lineJoin = 'round'
    g.beginPath(); g.moveTo(128, 14); g.lineTo(240, 224); g.lineTo(16, 224); g.closePath(); g.fill(); g.stroke()
    g.fillStyle = '#111'
    if (kind === 'uv') {
      g.font = 'bold 52px Helvetica, Arial'; g.textAlign = 'center'; g.fillText('UV', 128, 190)
      for (let i = -2; i <= 2; i++) { g.beginPath(); g.moveTo(128 + i * 18, 92); g.lineTo(128 + i * 30, 140); g.lineWidth = 7; g.stroke() }
      g.beginPath(); g.arc(128, 100, 14, 0, 7); g.fill()
    } else {
      g.beginPath(); g.arc(128, 150, 16, 0, 7); g.fill()
      for (let i = -3; i <= 3; i++) { g.beginPath(); g.moveTo(128, 150); g.lineTo(128 + i * 24, 70); g.lineWidth = 6; g.stroke() }
    }
    return finish(c, true)
  })
}

/** Breadboard top: 25 mm cell with an M6 tapped hole at the cell centre. `holes=false` gives a plain sheet. */
export function breadboardMaps(holes: boolean): { map: THREE.Texture; rough: THREE.Texture; bump: THREE.Texture } {
  const make = (kind: 'map' | 'rough' | 'bump') => memo(`bb-${holes}-${kind}`, () => {
    const N = 256
    const [c, g] = canvas(N, N)
    const base = kind === 'map' ? '#ffffff' : kind === 'rough' ? '#6e6e6e' : '#cfcfcf'
    g.fillStyle = base; g.fillRect(0, 0, N, N)
    if (kind !== 'map') {
      // fine blasted-metal grain
      let s = 777
      const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296)
      for (let i = 0; i < 2600; i++) {
        const v = kind === 'rough' ? 80 + rnd() * 90 : 170 + rnd() * 60
        g.fillStyle = `rgba(${v},${v},${v},0.5)`; g.fillRect(rnd() * N, rnd() * N, 1.3, 1.3)
      }
    }
    if (holes) {
      const r = (6 / 25) * N * 0.5          // M6 → Ø6 mm
      const cx = N / 2, cy = N / 2
      if (kind === 'map') {
        // chamfer ring then dark bore
        let gr = g.createRadialGradient(cx, cy, r * 0.7, cx, cy, r * 1.55)
        gr.addColorStop(0, 'rgba(255,255,255,0.0)'); gr.addColorStop(0.55, 'rgba(210,220,235,0.95)'); gr.addColorStop(1, 'rgba(255,255,255,0)')
        g.fillStyle = gr; g.beginPath(); g.arc(cx, cy, r * 1.55, 0, 7); g.fill()
        gr = g.createRadialGradient(cx - r * 0.25, cy - r * 0.25, 1, cx, cy, r)
        gr.addColorStop(0, '#2a2d33'); gr.addColorStop(0.7, '#050607'); gr.addColorStop(1, '#000')
        g.fillStyle = gr; g.beginPath(); g.arc(cx, cy, r, 0, 7); g.fill()
        // thread hint
        g.strokeStyle = 'rgba(140,150,165,0.35)'; g.lineWidth = 1
        for (let k = 1; k < 4; k++) { g.beginPath(); g.arc(cx, cy, r * (0.35 + k * 0.17), 0, 7); g.stroke() }
      } else if (kind === 'rough') {
        g.fillStyle = '#e0e0e0'; g.beginPath(); g.arc(cx, cy, r * 1.4, 0, 7); g.fill()
      } else {
        const gr = g.createRadialGradient(cx, cy, r * 0.8, cx, cy, r * 1.4)
        gr.addColorStop(0, '#000'); gr.addColorStop(1, 'rgba(207,207,207,0)')
        g.fillStyle = '#000'; g.beginPath(); g.arc(cx, cy, r, 0, 7); g.fill()
        g.fillStyle = gr; g.beginPath(); g.arc(cx, cy, r * 1.4, 0, 7); g.fill()
      }
    }
    // thin engraved cell grid lines are intentionally omitted (reads as noise at distance)
    const t = finish(c, kind === 'map', true)
    t.anisotropy = 16
    return t
  })
  return { map: make('map'), rough: make('rough'), bump: make('bump') }
}

/** Radial soft blob used for fake contact shadow beneath floating geometry. */
export function radialBlob(): THREE.Texture {
  return memo('blob', () => {
    const [c, g] = canvas(256, 256)
    const gr = g.createRadialGradient(128, 128, 0, 128, 128, 128)
    gr.addColorStop(0, 'rgba(0,0,0,0.85)'); gr.addColorStop(0.6, 'rgba(0,0,0,0.3)'); gr.addColorStop(1, 'rgba(0,0,0,0)')
    g.fillStyle = gr; g.fillRect(0, 0, 256, 256)
    return finish(c, true)
  })
}

/** Redrawn each time: the lamp's LCD. Returns a fresh CanvasTexture (caller owns it). */
export function drawLcd(tex: THREE.CanvasTexture | null, wavelengthNm: number | null, fwhm: number, name: string): THREE.CanvasTexture {
  const c = tex ? (tex.image as HTMLCanvasElement) : document.createElement('canvas')
  c.width = 512; c.height = 160
  const g = c.getContext('2d')!
  const gr = g.createLinearGradient(0, 0, 0, 160)
  gr.addColorStop(0, '#0b2a33'); gr.addColorStop(1, '#06161c')
  g.fillStyle = gr; g.fillRect(0, 0, 512, 160)
  g.fillStyle = 'rgba(80,220,255,0.06)'
  for (let y = 0; y < 160; y += 4) g.fillRect(0, y, 512, 1)
  g.textBaseline = 'alphabetic'
  g.fillStyle = '#6fe7ff'; g.font = '600 20px "Courier New", monospace'; g.textAlign = 'left'
  g.fillText(name.slice(0, 28).toUpperCase(), 16, 28)
  g.textAlign = 'right'; g.fillText(wavelengthNm == null ? 'SHUTTER CLOSED' : 'SHUTTER OPEN', 496, 28)
  g.shadowColor = '#3fd9ff'; g.shadowBlur = 14
  g.font = 'bold 84px "Courier New", monospace'; g.fillStyle = wavelengthNm == null ? '#2c6b7a' : '#b6f4ff'; g.textAlign = 'left'
  g.fillText(wavelengthNm == null ? '--- nm' : `${Math.round(wavelengthNm)} nm`, 16, 116)
  g.shadowBlur = 0
  g.font = '600 20px "Courier New", monospace'; g.fillStyle = '#6fe7ff'; g.textAlign = 'right'
  g.fillText(`BW ${fwhm.toFixed(0)} nm`, 496, 96); g.fillText('XENON 300 W', 496, 124)
  // bar-graph of lamp output
  for (let i = 0; i < 20; i++) { const h = 6 + 20 * Math.sin((i / 20) * Math.PI) ** 2; g.fillStyle = '#2fb7d6'; g.fillRect(16 + i * 12, 150 - h, 8, h) }
  const t = tex ?? new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  t.needsUpdate = true
  return t
}

/** Soft blotchy noise for bump-mapping turbid solids (lichen, tissue, paper fibres). */
export function noiseBump(): THREE.Texture {
  return memo('noise', () => {
    const [c, g] = canvas(256, 256)
    g.fillStyle = '#808080'; g.fillRect(0, 0, 256, 256)
    let s = 4242
    const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296)
    for (let i = 0; i < 900; i++) {
      const x = rnd() * 256, y = rnd() * 256, r = 2 + rnd() * 9, v = rnd() < 0.5 ? 40 : 220
      for (const [dx, dy] of [[0, 0], [256, 0], [-256, 0], [0, 256], [0, -256]]) {
        const gr = g.createRadialGradient(x + dx, y + dy, 0, x + dx, y + dy, r)
        gr.addColorStop(0, `rgba(${v},${v},${v},0.5)`); gr.addColorStop(1, `rgba(${v},${v},${v},0)`)
        g.fillStyle = gr; g.fillRect(x + dx - r, y + dy - r, r * 2, r * 2)
      }
    }
    return finish(c, false, true)
  })
}

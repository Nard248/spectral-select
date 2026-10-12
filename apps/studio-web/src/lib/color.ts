/** Wavelength (nm) → display colour. Visible range uses a smooth piecewise approximation of the
 * CIE response with intensity roll-off at the edges; UV (< 400 nm) maps to a deep violet "glow"
 * colour and NIR (> 700 nm) to a dark red so invisible light is still legible on screen. */
export function wavelengthToRGB(nm: number): [number, number, number] {
  let r = 0, g = 0, b = 0
  const l = Math.min(Math.max(nm, 380), 780)
  if (l < 440) { r = -(l - 440) / 60; b = 1 }
  else if (l < 490) { g = (l - 440) / 50; b = 1 }
  else if (l < 510) { g = 1; b = -(l - 510) / 20 }
  else if (l < 580) { r = (l - 510) / 70; g = 1 }
  else if (l < 645) { r = 1; g = -(l - 645) / 65 }
  else { r = 1 }
  let f = 1
  if (l < 420) f = 0.35 + 0.65 * (l - 380) / 40
  else if (l > 700) f = 0.35 + 0.65 * (780 - l) / 80
  if (nm < 380) { r = 0.55; g = 0.25; b = 1; f = 0.9 }   // UV: violet glow
  const gamma = 0.8
  const c = (v: number) => Math.pow(Math.max(v * f, 0), gamma)
  return [c(r), c(g), c(b)]
}

export function wavelengthToHex(nm: number): string {
  const [r, g, b] = wavelengthToRGB(nm)
  const h = (v: number) => Math.round(v * 255).toString(16).padStart(2, '0')
  return `#${h(r)}${h(g)}${h(b)}`
}

export const isInvisible = (nm: number) => nm < 380 || nm > 750

import { describe, it, expect } from 'vitest'
import { fmtSig, gtUnit } from './format'
describe('fmtSig', () => {
  it('keeps small values visible', () => { expect(fmtSig(6.5e-5)).toBe('6.50e-5'); expect(fmtSig(6.5e-5, 'mW')).toBe('6.50e-5 mW') })
  it('3 significant figures in the normal range', () => { expect(fmtSig(12.3456)).toBe('12.3'); expect(fmtSig(0.0123456)).toBe('0.0123'); expect(fmtSig(0)).toBe('0') })
  it('exponential for large values', () => { expect(fmtSig(2.5e6)).toBe('2.50e6'); expect(fmtSig(null)).toBe('—') })
  it('units from field names', () => { expect(gtUnit('depth_mm')).toBe('mm'); expect(gtUnit('incidence_deg')).toBe('°') })
})

import { describe, expect, it } from 'vitest'
import { describeProgress, etaSeconds, fmtEta } from './progress'

const wls = [310, 325, 340]
describe('describeProgress', () => {
  it('names the excitation in sensor messages', () => {
    const p = describeProgress('sensor 2/3', wls)
    expect(p.activeNm).toBe(325)
    expect(p.label).toContain('325 nm')
  })
  it('labels radiance blocks and cycles the beam colour', () => {
    const p = describeProgress('radiance 4/22', wls)
    expect(p.stage).toBe('transport')
    expect(p.label).toContain('block 4/22')
    expect(p.activeNm).toBe(310)
  })
  it('parses mock messages', () => expect(describeProgress('Exposing 405 nm (2/5)', wls).activeNm).toBe(405))
  it('passes unknown messages through', () => expect(describeProgress('weird', wls).label).toBe('Weird'))
})
describe('eta', () => {
  it('extrapolates from elapsed time', () => expect(etaSeconds(0.5, 0, 10_000)).toBeCloseTo(10))
  it('is null at the start', () => expect(etaSeconds(0.01, 0, 1000)).toBeNull())
  it('formats', () => { expect(fmtEta(12)).toBe('~12 s left'); expect(fmtEta(125)).toContain('min') })
})

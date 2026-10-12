let counter = 0
/** Unique, human-readable id: `<prefix>_<base36 time><counter>`. */
export function uid(prefix: string): string {
  counter = (counter + 1) % 1296
  return `${prefix}_${Date.now().toString(36).slice(-5)}${counter.toString(36).padStart(2, '0')}`
}
/** Next unused id of the form `<base>_<n>` among `taken`. */
export function uniqueId(base: string, taken: Iterable<string>): string {
  const set = new Set(taken)
  if (!set.has(base)) return base
  let n = 2
  while (set.has(`${base}_${n}`)) n++
  return `${base}_${n}`
}

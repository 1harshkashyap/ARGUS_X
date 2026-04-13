// ═══════════════════════════════════════════════════════════════════════
// ARGUS-X — Helper Utilities
// ═══════════════════════════════════════════════════════════════════════

let _uid = 0;

/** Generate a unique incrementing ID */
export function uid(): number {
  return ++_uid;
}

/** Random integer in [min, max] inclusive */
export function randInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

/** Random float in [min, max] with 2 decimal places */
export function randFloat(min: number, max: number): number {
  return +(Math.random() * (max - min) + min).toFixed(2);
}

/** Pick a random element from an array */
export function randItem<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

const GOLDEN_GAMMA = 0x9e37_79b9;
const ROTATE_BITS = 13;
const MULTIPLIER = 1_664_525;
const INCREMENT = 1_013_904_223;

export function fillSequence(values: Uint32Array, seed = 1): void {
  for (let index = 0; index < values.length; index += 1) {
    values[index] = (seed + index) >>> 0;
  }
}

export function mutateU32Array(values: Uint32Array, passes: number): number {
  let checksum = 0;

  for (let pass = 0; pass < passes; pass += 1) {
    for (let index = 0; index < values.length; index += 1) {
      const salt = Math.imul((pass + index) >>> 0, GOLDEN_GAMMA) >>> 0;
      const mixed = rotateLeft32((values[index] + salt) >>> 0, ROTATE_BITS);
      const next = (Math.imul(mixed, MULTIPLIER) + INCREMENT) >>> 0;

      values[index] = next;
      checksum = (checksum + next) >>> 0;
    }
  }

  return checksum >>> 0;
}

function rotateLeft32(value: number, bits: number): number {
  return ((value << bits) | (value >>> (32 - bits))) >>> 0;
}

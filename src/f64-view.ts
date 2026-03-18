const F64_BYTES = Float64Array.BYTES_PER_ELEMENT;

export function asF64View(buffer: Buffer, expectedCells?: number): Float64Array {
  if (buffer.byteLength % F64_BYTES !== 0) {
    throw new Error(
      `Expected a buffer length divisible by ${F64_BYTES}, got ${buffer.byteLength}`,
    );
  }

  const cells = buffer.byteLength / F64_BYTES;

  if (expectedCells !== undefined && cells !== expectedCells) {
    throw new Error(`Expected ${expectedCells} f64 values, got ${cells}`);
  }

  return new Float64Array(buffer.buffer, buffer.byteOffset, cells);
}

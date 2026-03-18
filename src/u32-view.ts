const U32_BYTES = Uint32Array.BYTES_PER_ELEMENT;

export function asU32View(buffer: Buffer, expectedCells?: number): Uint32Array {
  if (buffer.byteLength % U32_BYTES !== 0) {
    throw new Error(
      `Expected a buffer length divisible by ${U32_BYTES}, got ${buffer.byteLength}`,
    );
  }

  const cells = buffer.byteLength / U32_BYTES;

  if (expectedCells !== undefined && cells !== expectedCells) {
    throw new Error(`Expected ${expectedCells} u32 values, got ${cells}`);
  }

  return new Uint32Array(buffer.buffer, buffer.byteOffset, cells);
}

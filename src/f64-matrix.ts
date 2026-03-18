export type MatrixBatchAggregation = {
  averageColumnSums: Float64Array;
  grandTotal: number;
};

export function checkedMatrixCells(rows: number, cols: number): number {
  if (!Number.isInteger(rows) || rows <= 0) {
    throw new Error(`Expected a positive integer row count, got ${rows}`);
  }

  if (!Number.isInteger(cols) || cols <= 0) {
    throw new Error(`Expected a positive integer column count, got ${cols}`);
  }

  const cells = rows * cols;

  if (!Number.isSafeInteger(cells)) {
    throw new Error(`Matrix cell count overflows Number safety bounds: ${rows}x${cols}`);
  }

  return cells;
}

export function checkedMatrixBatchCells(
  matrices: number,
  rows: number,
  cols: number,
): number {
  if (!Number.isInteger(matrices) || matrices <= 0) {
    throw new Error(`Expected a positive integer matrix count, got ${matrices}`);
  }

  const cellsPerMatrix = checkedMatrixCells(rows, cols);
  const cells = matrices * cellsPerMatrix;

  if (!Number.isSafeInteger(cells)) {
    throw new Error(
      `Matrix batch cell count overflows Number safety bounds: ${matrices} x ${rows} x ${cols}`,
    );
  }

  return cells;
}

export function matrixValue(matrix: number, row: number, col: number): number {
  if (!Number.isInteger(matrix) || matrix < 0) {
    throw new Error(`Expected a non-negative integer matrix index, got ${matrix}`);
  }

  if (!Number.isInteger(row) || row < 0) {
    throw new Error(`Expected a non-negative integer row index, got ${row}`);
  }

  if (!Number.isInteger(col) || col < 0) {
    throw new Error(`Expected a non-negative integer column index, got ${col}`);
  }

  return matrix * 0.75 + row * 0.5 + col * 0.25 + ((matrix ^ row ^ col) & 7) * 0.125;
}

export function fillF64Matrices(
  values: Float64Array,
  matrices: number,
  rows: number,
  cols: number,
): void {
  const cells = checkedMatrixBatchCells(matrices, rows, cols);
  const cellsPerMatrix = checkedMatrixCells(rows, cols);

  if (values.length !== cells) {
    throw new Error(
      `Expected exactly ${cells} f64 values for ${matrices} matrices of shape ${rows}x${cols}, got ${values.length}`,
    );
  }

  for (let matrix = 0; matrix < matrices; matrix += 1) {
    const matrixOffset = matrix * cellsPerMatrix;

    for (let row = 0; row < rows; row += 1) {
      const rowStart = matrixOffset + row * cols;

      for (let col = 0; col < cols; col += 1) {
        values[rowStart + col] = matrixValue(matrix, row, col);
      }
    }
  }
}

export function aggregateMatricesAverageColumnsAndGrandTotalInto(
  values: Float64Array,
  matrices: number,
  rows: number,
  cols: number,
  averageColumnSums: Float64Array,
): number {
  const cells = checkedMatrixBatchCells(matrices, rows, cols);
  const cellsPerMatrix = checkedMatrixCells(rows, cols);

  if (values.length !== cells) {
    throw new Error(
      `Expected exactly ${cells} f64 values for ${matrices} matrices of shape ${rows}x${cols}, got ${values.length}`,
    );
  }

  if (averageColumnSums.length !== cols) {
    throw new Error(
      `Expected exactly ${cols} averaged column sums, got ${averageColumnSums.length}`,
    );
  }

  averageColumnSums.fill(0);
  const matrixColumnSums = new Float64Array(cols);
  let grandTotal = 0;

  for (let matrix = 0; matrix < matrices; matrix += 1) {
    matrixColumnSums.fill(0);
    const matrixOffset = matrix * cellsPerMatrix;

    for (let row = 0; row < rows; row += 1) {
      const rowStart = matrixOffset + row * cols;

      for (let col = 0; col < cols; col += 1) {
        matrixColumnSums[col] += values[rowStart + col];
      }
    }

    let matrixTotal = 0;

    for (let col = 0; col < cols; col += 1) {
      averageColumnSums[col] += matrixColumnSums[col];
      matrixTotal += matrixColumnSums[col];
    }

    grandTotal += matrixTotal;
  }

  for (let col = 0; col < cols; col += 1) {
    averageColumnSums[col] /= matrices;
  }

  return grandTotal;
}

export function aggregateMatricesAverageColumnsAndGrandTotal(
  values: Float64Array,
  matrices: number,
  rows: number,
  cols: number,
): MatrixBatchAggregation {
  const averageColumnSums = new Float64Array(cols);
  const grandTotal = aggregateMatricesAverageColumnsAndGrandTotalInto(
    values,
    matrices,
    rows,
    cols,
    averageColumnSums,
  );

  return { averageColumnSums, grandTotal };
}

export type ColumnAggregation = {
  columnSums: Float64Array;
  total: number;
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

export function matrixValue(row: number, col: number): number {
  if (!Number.isInteger(row) || row < 0) {
    throw new Error(`Expected a non-negative integer row index, got ${row}`);
  }

  if (!Number.isInteger(col) || col < 0) {
    throw new Error(`Expected a non-negative integer column index, got ${col}`);
  }

  return row * 0.5 + col * 0.25 + ((row ^ col) & 7) * 0.125;
}

export function fillF64Matrix(
  values: Float64Array,
  rows: number,
  cols: number,
): void {
  const cells = checkedMatrixCells(rows, cols);

  if (values.length !== cells) {
    throw new Error(
      `Expected exactly ${cells} f64 values for a ${rows}x${cols} matrix, got ${values.length}`,
    );
  }

  for (let row = 0; row < rows; row += 1) {
    const rowStart = row * cols;

    for (let col = 0; col < cols; col += 1) {
      values[rowStart + col] = matrixValue(row, col);
    }
  }
}

export function aggregateColumnsAndTotalInto(
  values: Float64Array,
  rows: number,
  cols: number,
  columnSums: Float64Array,
): number {
  const cells = checkedMatrixCells(rows, cols);

  if (values.length !== cells) {
    throw new Error(
      `Expected exactly ${cells} f64 values for a ${rows}x${cols} matrix, got ${values.length}`,
    );
  }

  if (columnSums.length !== cols) {
    throw new Error(`Expected exactly ${cols} column sums, got ${columnSums.length}`);
  }

  columnSums.fill(0);

  for (let row = 0; row < rows; row += 1) {
    const rowStart = row * cols;

    for (let col = 0; col < cols; col += 1) {
      columnSums[col] += values[rowStart + col];
    }
  }

  let total = 0;

  for (let col = 0; col < cols; col += 1) {
    total += columnSums[col];
  }

  return total;
}

export function aggregateColumnsAndTotal(
  values: Float64Array,
  rows: number,
  cols: number,
): ColumnAggregation {
  const columnSums = new Float64Array(cols);
  const total = aggregateColumnsAndTotalInto(values, rows, cols, columnSums);

  return { columnSums, total };
}

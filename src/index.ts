import {
  aggregateMatricesAverageColumnsAndGrandTotal,
  aggregateMatricesAverageColumnsAndGrandTotalInto,
  checkedMatrixBatchCells,
  checkedMatrixCells,
  fillF64Matrices,
  matrixValue,
  type MatrixBatchAggregation,
} from './f64-matrix';
import { asF64View } from './f64-view';
import { fillSequence, mutateU32Array } from './mutation';
import { asU32View } from './u32-view';

const MATRIX_DIMENSION = 6;
const MATRIX_CELLS = MATRIX_DIMENSION * MATRIX_DIMENSION;
const U32_BYTES = Uint32Array.BYTES_PER_ELEMENT;
const MATRIX_BYTES = MATRIX_CELLS * U32_BYTES;

type NativeBindings = {
  createSharedMatrix(): Buffer;
  createU32Buffer(cells: number): Buffer;
  createF64Buffer(cells: number): Buffer;
  createF64Matrices(matrices: number, rows: number, cols: number): Buffer;
  mutateU32Buffer(buffer: Buffer, passes: number): number;
  printMatrix6x6(buffer: Buffer): void;
  fillF64MatricesBuffer(
    buffer: Buffer,
    matrices: number,
    rows: number,
    cols: number,
  ): void;
  f64MatrixValue(matrix: number, row: number, col: number): number;
  aggregateF64MatrixBatch(
    matricesBuffer: Buffer,
    matrices: number,
    rows: number,
    cols: number,
    averageColumnSumsBuffer: Buffer,
  ): number;
  aggregateF64MatrixBatchAllocating(
    matricesBuffer: Buffer,
    matrices: number,
    rows: number,
    cols: number,
  ): NativeMatrixBatchAggregationResult;
  sampleGpuMap2SourceCode(): string;
  compileGpuMap2Pipeline(source: string): GpuCompilationArtifact;
  compileSampleGpuMap2(): GpuCompilationArtifact;
  sampleMatrixReductionSourceCode(): string;
  compileMatrixReductionGpuPipeline(
    matrices: number,
    rows: number,
    cols: number,
  ): MatrixReductionGpuPipelineArtifact;
};

const native = require('../native/index.node') as NativeBindings;

type NativeMatrixBatchAggregationResult = {
  averageColumnSums: Buffer;
  grandTotal: number;
};

export type GpuCompilationArtifact = {
  source: string;
  jsAst: string;
  kernelIr: string;
  ptx: string;
  hostLaunch: string;
  notes: string[];
};

export type MatrixReductionGpuPipelineArtifact = {
  source: string;
  reductionIr: string;
  stage1KernelIr: string;
  stage2KernelIr: string;
  stage3KernelIr: string;
  stage4KernelIr: string;
  ptx: string;
  hostLaunch: string;
  notes: string[];
};

export function createSharedMatrix(): Buffer {
  return native.createSharedMatrix();
}

export function createSharedU32Buffer(cells: number): Buffer {
  if (!Number.isInteger(cells) || cells < 0) {
    throw new Error(`Expected a non-negative integer cell count, got ${cells}`);
  }

  return native.createU32Buffer(cells);
}

export function createSharedF64Buffer(cells: number): Buffer {
  if (!Number.isInteger(cells) || cells < 0) {
    throw new Error(`Expected a non-negative integer cell count, got ${cells}`);
  }

  return native.createF64Buffer(cells);
}

export function createSharedF64Matrices(
  matrices: number,
  rows: number,
  cols: number,
): Buffer {
  checkedMatrixBatchCells(matrices, rows, cols);
  return native.createF64Matrices(matrices, rows, cols);
}

export function asMatrixView(buffer: Buffer): Uint32Array {
  if (buffer.byteLength !== MATRIX_BYTES) {
    throw new Error(
      `Expected a ${MATRIX_BYTES}-byte buffer for a 6x6 u32 matrix, got ${buffer.byteLength}`,
    );
  }

  return asU32View(buffer, MATRIX_CELLS);
}

export function createInitializedMatrix6x6(): Buffer {
  const buffer = createSharedMatrix();
  const values = asMatrixView(buffer);
  fillSequence(values, 1);
  return buffer;
}

export function printMatrix6x6(values: ArrayLike<number>): void {
  if (values.length !== MATRIX_CELLS) {
    throw new Error(
      `Expected ${MATRIX_CELLS} values for a 6x6 matrix, got ${values.length}`,
    );
  }

  for (let row = 0; row < MATRIX_DIMENSION; row += 1) {
    const start = row * MATRIX_DIMENSION;
    const line = Array.from({ length: MATRIX_DIMENSION }, (_, column) =>
      values[start + column].toString().padStart(4, ' '),
    ).join(' ');

    console.log(line);
  }
}

export function mutateSharedU32Buffer(buffer: Buffer, passes: number): number {
  return native.mutateU32Buffer(buffer, passes);
}

export function mutateMatrix6x6InJs(buffer: Buffer, passes: number): number {
  return mutateU32Array(asMatrixView(buffer), passes);
}

export function printNativeMatrix6x6(buffer: Buffer): void {
  native.printMatrix6x6(buffer);
}

export function fillSharedF64Matrices(
  buffer: Buffer,
  matrices: number,
  rows: number,
  cols: number,
): void {
  checkedMatrixBatchCells(matrices, rows, cols);
  native.fillF64MatricesBuffer(buffer, matrices, rows, cols);
}

export function nativeF64MatrixValue(
  matrix: number,
  row: number,
  col: number,
): number {
  return native.f64MatrixValue(matrix, row, col);
}

export function aggregateSharedF64MatricesInRust(
  matricesBuffer: Buffer,
  matrices: number,
  rows: number,
  cols: number,
  averageColumnSumsBuffer: Buffer,
): number {
  checkedMatrixBatchCells(matrices, rows, cols);
  return native.aggregateF64MatrixBatch(
    matricesBuffer,
    matrices,
    rows,
    cols,
    averageColumnSumsBuffer,
  );
}

export function aggregateSharedF64MatricesInRustAllocating(
  matricesBuffer: Buffer,
  matrices: number,
  rows: number,
  cols: number,
): MatrixBatchAggregation {
  checkedMatrixBatchCells(matrices, rows, cols);
  const result = native.aggregateF64MatrixBatchAllocating(
    matricesBuffer,
    matrices,
    rows,
    cols,
  );

  return {
    averageColumnSums: asF64View(result.averageColumnSums, cols),
    grandTotal: result.grandTotal,
  };
}

export function sampleGpuMap2SourceCode(): string {
  return native.sampleGpuMap2SourceCode();
}

export function compileGpuMap2Pipeline(source: string): GpuCompilationArtifact {
  return native.compileGpuMap2Pipeline(source);
}

export function compileSampleGpuMap2(): GpuCompilationArtifact {
  return native.compileSampleGpuMap2();
}

export function sampleMatrixReductionSourceCode(): string {
  return native.sampleMatrixReductionSourceCode();
}

export function compileMatrixReductionGpuPipeline(
  matrices: number,
  rows: number,
  cols: number,
): MatrixReductionGpuPipelineArtifact {
  checkedMatrixBatchCells(matrices, rows, cols);
  return native.compileMatrixReductionGpuPipeline(matrices, rows, cols);
}

export {
  aggregateMatricesAverageColumnsAndGrandTotal,
  aggregateMatricesAverageColumnsAndGrandTotalInto,
  asF64View,
  asU32View,
  checkedMatrixBatchCells,
  checkedMatrixCells,
  fillF64Matrices,
  fillSequence,
  matrixValue,
  mutateU32Array,
  MATRIX_BYTES,
  MATRIX_CELLS,
  MATRIX_DIMENSION,
};

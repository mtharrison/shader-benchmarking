import {
  aggregateMatricesAverageColumnsAndGrandTotal,
  aggregateSharedF64MatricesInRustAllocating,
  asF64View,
  checkedMatrixBatchCells,
  compileMatrixReductionGpuPipeline,
  createSharedF64Buffer,
  fillF64Matrices,
  matrixValue,
  nativeF64MatrixValue,
  sampleMatrixReductionSourceCode,
} from './index';
import { tryRunCudaMatrixBenchmark } from './cuda-matrix-runner';

const MATRICES = 4;
const ROWS = 1_000;
const COLS = 1_000;
const PREVIEW_COLUMNS = 5;

function printSection(title: string, body: string): void {
  console.log(`\n=== ${title} ===`);
  console.log(body);
}

function formatPreview(values: Float64Array): string {
  return Array.from(values.slice(0, PREVIEW_COLUMNS))
    .map((value, index) => `col[${index}] = ${value.toFixed(3)}`)
    .join('\n');
}

function assertClose(actual: number, expected: number, label: string): void {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: ${actual} !== ${expected}`);
  }
}

const batchCells = checkedMatrixBatchCells(MATRICES, ROWS, COLS);
const matricesBuffer = createSharedF64Buffer(batchCells);
const matrices = asF64View(matricesBuffer, batchCells);
fillF64Matrices(matrices, MATRICES, ROWS, COLS);

const jsResult = aggregateMatricesAverageColumnsAndGrandTotal(
  matrices,
  MATRICES,
  ROWS,
  COLS,
);
const rustResult = aggregateSharedF64MatricesInRustAllocating(
  matricesBuffer,
  MATRICES,
  ROWS,
  COLS,
);
const gpuPipeline = compileMatrixReductionGpuPipeline(MATRICES, ROWS, COLS);
const gpuRun = tryRunCudaMatrixBenchmark({
  matrices: MATRICES,
  rows: ROWS,
  cols: COLS,
  warmup: 0,
  samples: 1,
});

assertClose(rustResult.grandTotal, jsResult.grandTotal, 'grand total');

for (let index = 0; index < COLS; index += 1) {
  assertClose(
    rustResult.averageColumnSums[index],
    jsResult.averageColumnSums[index],
    `average column ${index}`,
  );
}

assertClose(
  nativeF64MatrixValue(3, 12, 34),
  matrixValue(3, 12, 34),
  'matrix value formula',
);

printSection(
  'Matrix Batch',
  `${MATRICES} matrices of shape ${ROWS} x ${COLS} (${batchCells.toLocaleString()} cells)`,
);
printSection(
  'Sample Values',
  [
    `matrix[0,0,0] = ${matrixValue(0, 0, 0)}`,
    `matrix[3,12,34] = ${matrixValue(3, 12, 34)}`,
  ].join('\n'),
);
printSection('JavaScript Grand Total', jsResult.grandTotal.toFixed(3));
printSection('Average Column Preview', formatPreview(jsResult.averageColumnSums));
printSection('Rust Grand Total', rustResult.grandTotal.toFixed(3));
printSection('GPU Source', sampleMatrixReductionSourceCode());
printSection('GPU Reduction IR', gpuPipeline.reductionIr);
printSection('GPU Stage 1 Kernel', gpuPipeline.stage1KernelIr);
printSection('GPU Stage 2 Kernel', gpuPipeline.stage2KernelIr);
printSection('GPU Stage 3 Kernel', gpuPipeline.stage3KernelIr);
printSection('GPU Stage 4 Kernel', gpuPipeline.stage4KernelIr);
printSection('GPU Host Launch Sketch', gpuPipeline.hostLaunch);
printSection('GPU Notes', gpuPipeline.notes.map((note) => `- ${note}`).join('\n'));

if (gpuRun) {
  printSection(
    'GPU Runtime',
    [
      `device = ${gpuRun.deviceName}`,
      `avg_ms = ${gpuRun.averageMs.toFixed(3)}`,
      `grand_total = ${gpuRun.grandTotal.toFixed(3)}`,
      `preview = ${gpuRun.averageColumnPreview.map((value) => value.toFixed(3)).join(', ')}`,
    ].join('\n'),
  );
}

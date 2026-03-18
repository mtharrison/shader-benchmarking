import {
  aggregateColumnsAndTotal,
  aggregateSharedF64MatrixInRust,
  asF64View,
  compileMatrixReductionGpuPipeline,
  createSharedF64Buffer,
  fillF64Matrix,
  matrixValue,
  nativeF64MatrixValue,
  sampleMatrixReductionSourceCode,
} from './index';
import { tryRunCudaMatrixBenchmark } from './cuda-matrix-runner';

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

const cells = ROWS * COLS;
const matrixBuffer = createSharedF64Buffer(cells);
const matrix = asF64View(matrixBuffer, cells);
fillF64Matrix(matrix, ROWS, COLS);

const jsResult = aggregateColumnsAndTotal(matrix, ROWS, COLS);
const rustResult = aggregateSharedF64MatrixInRust(matrixBuffer, ROWS, COLS);
const gpuPipeline = compileMatrixReductionGpuPipeline(ROWS, COLS);
const gpuRun = tryRunCudaMatrixBenchmark({
  rows: ROWS,
  cols: COLS,
  warmup: 0,
  samples: 1,
});

assertClose(rustResult.total, jsResult.total, 'total');

for (let index = 0; index < COLS; index += 1) {
  assertClose(rustResult.columnSums[index], jsResult.columnSums[index], `column ${index}`);
}

assertClose(nativeF64MatrixValue(12, 34), matrixValue(12, 34), 'matrix value formula');

printSection('Matrix', `${ROWS} x ${COLS} float64 values (${cells.toLocaleString()} cells)`);
printSection(
  'Sample Values',
  [`matrix[0,0] = ${matrixValue(0, 0)}`, `matrix[12,34] = ${matrixValue(12, 34)}`].join('\n'),
);
printSection('JavaScript Total', jsResult.total.toFixed(3));
printSection('Column Sum Preview', formatPreview(jsResult.columnSums));
printSection('Rust Total', rustResult.total.toFixed(3));
printSection('GPU Source', sampleMatrixReductionSourceCode());
printSection('GPU Reduction IR', gpuPipeline.reductionIr);
printSection('GPU Stage 1 Kernel', gpuPipeline.stage1KernelIr);
printSection('GPU Stage 2 Kernel', gpuPipeline.stage2KernelIr);
printSection('GPU Host Launch Sketch', gpuPipeline.hostLaunch);
printSection('GPU Notes', gpuPipeline.notes.map((note) => `- ${note}`).join('\n'));

if (gpuRun) {
  printSection(
    'GPU Runtime',
    [
      `device = ${gpuRun.deviceName}`,
      `avg_ms = ${gpuRun.averageMs.toFixed(3)}`,
      `total = ${gpuRun.total.toFixed(3)}`,
      `preview = ${gpuRun.columnPreview.map((value) => value.toFixed(3)).join(', ')}`,
    ].join('\n'),
  );
}

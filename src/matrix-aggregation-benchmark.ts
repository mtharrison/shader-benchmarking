import {
  aggregateColumnsAndTotalInto,
  aggregateSharedF64MatrixColumnsInRust,
  asF64View,
  compileMatrixReductionGpuPipeline,
  createSharedF64Buffer,
  fillF64Matrix,
  checkedMatrixCells,
} from './index';
import { tryRunCudaMatrixBenchmark } from './cuda-matrix-runner';

type BenchmarkOptions = {
  rows: number;
  cols: number;
  warmup: number;
  samples: number;
};

type BenchmarkTask = {
  name: string;
  run: () => number;
  readColumnSums: () => Float64Array;
};

type BenchmarkResult = {
  name: string;
  total: number;
  averageMs: number;
  medianMs: number;
  minMs: number;
  maxMs: number;
  gibPerSecond: number;
};

const DEFAULT_ROWS = 1_000;
const DEFAULT_COLS = 1_000;
const DEFAULT_WARMUP = 3;
const DEFAULT_SAMPLES = 10;
const PREVIEW_COLUMNS = 5;

function main(): void {
  const options = parseArgs(process.argv.slice(2));
  const cells = checkedMatrixCells(options.rows, options.cols);
  const matrixBytes = cells * Float64Array.BYTES_PER_ELEMENT;
  const columnBytes = options.cols * Float64Array.BYTES_PER_ELEMENT;
  const touchedBytes = matrixBytes + (2 * columnBytes);

  const matrixBuffer = createSharedF64Buffer(cells);
  const matrix = asF64View(matrixBuffer, cells);
  fillF64Matrix(matrix, options.rows, options.cols);

  const jsColumnSums = new Float64Array(options.cols);
  const rustColumnSumsBuffer = createSharedF64Buffer(options.cols);
  const rustColumnSums = asF64View(rustColumnSumsBuffer, options.cols);

  const referenceTotal = aggregateColumnsAndTotalInto(
    matrix,
    options.rows,
    options.cols,
    jsColumnSums,
  );
  const referenceColumnSums = Float64Array.from(jsColumnSums);

  const rustTotal = aggregateSharedF64MatrixColumnsInRust(
    matrixBuffer,
    options.rows,
    options.cols,
    rustColumnSumsBuffer,
  );

  assertSameResults(referenceTotal, referenceColumnSums, rustTotal, rustColumnSums, 'rust / napi');

  const tasks: BenchmarkTask[] = [
    {
      name: 'js / float64 column aggregation',
      run: () =>
        aggregateColumnsAndTotalInto(matrix, options.rows, options.cols, jsColumnSums),
      readColumnSums: () => jsColumnSums,
    },
    {
      name: 'rust / napi column aggregation',
      run: () =>
        aggregateSharedF64MatrixColumnsInRust(
          matrixBuffer,
          options.rows,
          options.cols,
          rustColumnSumsBuffer,
        ),
      readColumnSums: () => rustColumnSums,
    },
  ];

  console.log(
    [
      `Benchmarking ${options.rows}x${options.cols} float64 matrix aggregation`,
      `${cells.toLocaleString()} cells`,
      `${formatMiB(matrixBytes)} matrix`,
      `warmup=${options.warmup}`,
      `samples=${options.samples}`,
    ].join('  '),
  );
  console.log(
    `Each run computes ${options.cols.toLocaleString()} column sums and then a total from those sums.`,
  );
  console.log(
    'If CUDA is available, the benchmark also runs the GPU implementation; otherwise it logs an error and continues.',
  );
  console.log('');

  const results = tasks.map((task) =>
    runTask(task, options, touchedBytes, referenceTotal, referenceColumnSums),
  );

  const gpuResult = tryRunCudaMatrixBenchmark({
    rows: options.rows,
    cols: options.cols,
    warmup: options.warmup,
    samples: options.samples,
  });

  if (gpuResult) {
    if (gpuResult.total !== referenceTotal) {
      throw new Error(`gpu total mismatch: ${gpuResult.total} !== ${referenceTotal}`);
    }

    const preview = referenceColumnSums.slice(0, gpuResult.columnPreview.length);
    for (let index = 0; index < gpuResult.columnPreview.length; index += 1) {
      if (gpuResult.columnPreview[index] !== preview[index]) {
        throw new Error(
          `gpu preview mismatch at column ${index}: ${gpuResult.columnPreview[index]} !== ${preview[index]}`,
        );
      }
    }

    results.push({
      name: gpuResult.name,
      total: gpuResult.total,
      averageMs: gpuResult.averageMs,
      medianMs: gpuResult.medianMs,
      minMs: gpuResult.minMs,
      maxMs: gpuResult.maxMs,
      gibPerSecond: gpuResult.gibPerSecond,
    });
  }

  printResults(results);

  const gpuPipeline = compileMatrixReductionGpuPipeline(options.rows, options.cols);
  console.log('');
  if (gpuResult) {
    console.log(`GPU device: ${gpuResult.deviceName}`);
  }
  console.log(`GPU lowering artifact: ${gpuPipeline.stage1KernelIr.split('\n', 1)[0]}`);
  console.log(`GPU lowering artifact: ${gpuPipeline.stage2KernelIr.split('\n', 1)[0]}`);
  console.log(`GPU PTX size: ${gpuPipeline.ptx.length.toLocaleString()} chars`);
  console.log(`Reference total: ${referenceTotal.toFixed(3)}`);
  console.log(
    `Reference columns: ${Array.from(referenceColumnSums.slice(0, PREVIEW_COLUMNS))
      .map((value) => value.toFixed(3))
      .join(', ')}`,
  );
}

function runTask(
  task: BenchmarkTask,
  options: BenchmarkOptions,
  touchedBytes: number,
  referenceTotal: number,
  referenceColumnSums: Float64Array,
): BenchmarkResult {
  for (let iteration = 0; iteration < options.warmup; iteration += 1) {
    task.run();
  }

  const samples: number[] = [];
  let total = 0;

  for (let iteration = 0; iteration < options.samples; iteration += 1) {
    const startedAt = process.hrtime.bigint();
    total = task.run();
    const elapsedMs = Number(process.hrtime.bigint() - startedAt) / 1_000_000;
    samples.push(elapsedMs);
  }

  assertSameResults(
    referenceTotal,
    referenceColumnSums,
    total,
    task.readColumnSums(),
    task.name,
  );

  const sorted = [...samples].sort((left, right) => left - right);
  const averageMs = samples.reduce((sum, sample) => sum + sample, 0) / samples.length;
  const medianMs = sorted[Math.floor(sorted.length / 2)];
  const minMs = sorted[0];
  const maxMs = sorted[sorted.length - 1];
  const gibPerSecond = touchedBytes / (1024 ** 3) / (averageMs / 1_000);

  return {
    name: task.name,
    total,
    averageMs,
    medianMs,
    minMs,
    maxMs,
    gibPerSecond,
  };
}

function assertSameResults(
  expectedTotal: number,
  expectedColumnSums: Float64Array,
  actualTotal: number,
  actualColumnSums: Float64Array,
  label: string,
): void {
  if (actualTotal !== expectedTotal) {
    throw new Error(`${label} total mismatch: ${actualTotal} !== ${expectedTotal}`);
  }

  if (actualColumnSums.length !== expectedColumnSums.length) {
    throw new Error(
      `${label} column count mismatch: ${actualColumnSums.length} !== ${expectedColumnSums.length}`,
    );
  }

  for (let index = 0; index < expectedColumnSums.length; index += 1) {
    if (actualColumnSums[index] !== expectedColumnSums[index]) {
      throw new Error(
        `${label} column ${index} mismatch: ${actualColumnSums[index]} !== ${expectedColumnSums[index]}`,
      );
    }
  }
}

function printResults(results: BenchmarkResult[]): void {
  const fastestAverage = Math.min(...results.map((result) => result.averageMs));

  for (const result of results) {
    const relative = result.averageMs / fastestAverage;
    console.log(
      [
        result.name.padEnd(34),
        `avg ${result.averageMs.toFixed(3).padStart(9)} ms`,
        `med ${result.medianMs.toFixed(3).padStart(9)} ms`,
        `min ${result.minMs.toFixed(3).padStart(9)} ms`,
        `max ${result.maxMs.toFixed(3).padStart(9)} ms`,
        `${result.gibPerSecond.toFixed(2).padStart(6)} GiB/s`,
        `${relative.toFixed(2).padStart(5)}x`,
      ].join('  '),
    );
  }
}

function formatMiB(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
}

function parseArgs(argv: string[]): BenchmarkOptions {
  const options: BenchmarkOptions = {
    rows: DEFAULT_ROWS,
    cols: DEFAULT_COLS,
    warmup: DEFAULT_WARMUP,
    samples: DEFAULT_SAMPLES,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];

    switch (arg) {
      case '--rows':
        options.rows = parsePositiveInt(arg, next);
        index += 1;
        break;
      case '--cols':
        options.cols = parsePositiveInt(arg, next);
        index += 1;
        break;
      case '--warmup':
        options.warmup = parseNonNegativeInt(arg, next);
        index += 1;
        break;
      case '--samples':
        options.samples = parsePositiveInt(arg, next);
        index += 1;
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  checkedMatrixCells(options.rows, options.cols);

  return options;
}

function parsePositiveInt(flag: string, value: string | undefined): number {
  if (value === undefined) {
    throw new Error(`Missing value for ${flag}`);
  }

  const parsed = Number.parseInt(value, 10);

  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`Expected ${flag} to be a positive integer, got ${value}`);
  }

  return parsed;
}

function parseNonNegativeInt(flag: string, value: string | undefined): number {
  if (value === undefined) {
    throw new Error(`Missing value for ${flag}`);
  }

  const parsed = Number.parseInt(value, 10);

  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`Expected ${flag} to be a non-negative integer, got ${value}`);
  }

  return parsed;
}

main();

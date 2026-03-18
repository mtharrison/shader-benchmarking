import {
  aggregateMatricesAverageColumnsAndGrandTotalInto,
  aggregateSharedF64MatricesInRust,
  aggregateSharedF64MatricesInRustParallel,
  asF64View,
  checkedMatrixBatchCells,
  compileMatrixReductionGpuPipeline,
  createSharedF64Buffer,
  fillF64Matrices,
} from './index';
import { tryRunCudaMatrixBenchmarks } from './cuda-matrix-runner';

type BenchmarkOptions = {
  matrices: number;
  rows: number;
  cols: number;
  warmup: number;
  samples: number;
};

type BenchmarkTask = {
  name: string;
  run: () => number;
  readAverageColumnSums: () => Float64Array;
};

type BenchmarkResult = {
  name: string;
  grandTotal: number;
  averageMs: number;
  medianMs: number;
  minMs: number;
  maxMs: number;
  gibPerSecond: number;
};

const DEFAULT_MATRICES = 4;
const DEFAULT_ROWS = 1_000;
const DEFAULT_COLS = 1_000;
const DEFAULT_WARMUP = 3;
const DEFAULT_SAMPLES = 10;
const PREVIEW_COLUMNS = 5;

function main(): void {
  const options = parseArgs(process.argv.slice(2));
  const cells = checkedMatrixBatchCells(options.matrices, options.rows, options.cols);
  const matrixBytes = cells * Float64Array.BYTES_PER_ELEMENT;
  const averageColumnBytes = options.cols * Float64Array.BYTES_PER_ELEMENT;
  const touchedBytes = matrixBytes + averageColumnBytes + Float64Array.BYTES_PER_ELEMENT;

  const matricesBuffer = createSharedF64Buffer(cells);
  const matrices = asF64View(matricesBuffer, cells);
  fillF64Matrices(matrices, options.matrices, options.rows, options.cols);

  const jsAverageColumnSums = new Float64Array(options.cols);
  const rustAverageColumnSumsBuffer = createSharedF64Buffer(options.cols);
  const rustAverageColumnSums = asF64View(rustAverageColumnSumsBuffer, options.cols);
  const rustParallelAverageColumnSumsBuffer = createSharedF64Buffer(options.cols);
  const rustParallelAverageColumnSums = asF64View(
    rustParallelAverageColumnSumsBuffer,
    options.cols,
  );

  const referenceGrandTotal = aggregateMatricesAverageColumnsAndGrandTotalInto(
    matrices,
    options.matrices,
    options.rows,
    options.cols,
    jsAverageColumnSums,
  );
  const referenceAverageColumnSums = Float64Array.from(jsAverageColumnSums);

  const rustGrandTotal = aggregateSharedF64MatricesInRust(
    matricesBuffer,
    options.matrices,
    options.rows,
    options.cols,
    rustAverageColumnSumsBuffer,
  );
  const rustParallelGrandTotal = aggregateSharedF64MatricesInRustParallel(
    matricesBuffer,
    options.matrices,
    options.rows,
    options.cols,
    rustParallelAverageColumnSumsBuffer,
  );

  assertSameResults(
    referenceGrandTotal,
    referenceAverageColumnSums,
    rustGrandTotal,
    rustAverageColumnSums,
    'rust / napi',
  );
  assertSameResults(
    referenceGrandTotal,
    referenceAverageColumnSums,
    rustParallelGrandTotal,
    rustParallelAverageColumnSums,
    'rust / napi parallel',
  );

  const tasks: BenchmarkTask[] = [
    {
      name: 'js / f64 batch aggregation',
      run: () =>
        aggregateMatricesAverageColumnsAndGrandTotalInto(
          matrices,
          options.matrices,
          options.rows,
          options.cols,
          jsAverageColumnSums,
        ),
      readAverageColumnSums: () => jsAverageColumnSums,
    },
    {
      name: 'rust / napi batch aggregation',
      run: () =>
        aggregateSharedF64MatricesInRust(
          matricesBuffer,
          options.matrices,
          options.rows,
          options.cols,
          rustAverageColumnSumsBuffer,
        ),
      readAverageColumnSums: () => rustAverageColumnSums,
    },
    {
      name: 'rust / napi parallel batch aggregation',
      run: () =>
        aggregateSharedF64MatricesInRustParallel(
          matricesBuffer,
          options.matrices,
          options.rows,
          options.cols,
          rustParallelAverageColumnSumsBuffer,
        ),
      readAverageColumnSums: () => rustParallelAverageColumnSums,
    },
  ];

  console.log(
    [
      `Benchmarking ${options.matrices} matrices of shape ${options.rows}x${options.cols}`,
      `${cells.toLocaleString()} f64 cells`,
      `${formatMiB(matrixBytes)} batch`,
      `warmup=${options.warmup}`,
      `samples=${options.samples}`,
    ].join('  '),
  );
  console.log(
    `Each run computes per-matrix column sums, averages those columns across ${options.matrices} matrices, and accumulates a grand total from the per-matrix totals.`,
  );
  console.log(
    'If CUDA is available, the benchmark also runs explicit `e2e pageable`, `e2e pinned`, and `resident` cuBLAS-backed GPU rows; otherwise it logs an error and continues.',
  );
  console.log('');

  const results = tasks.map((task) =>
    runTask(task, options, touchedBytes, referenceGrandTotal, referenceAverageColumnSums),
  );

  const gpuResults = tryRunCudaMatrixBenchmarks({
    matrices: options.matrices,
    rows: options.rows,
    cols: options.cols,
    warmup: options.warmup,
    samples: options.samples,
  });

  if (gpuResults) {
    for (const gpuResult of gpuResults) {
      if (gpuResult.grandTotal !== referenceGrandTotal) {
        throw new Error(
          `gpu grand total mismatch: ${gpuResult.grandTotal} !== ${referenceGrandTotal}`,
        );
      }

      const preview = referenceAverageColumnSums.slice(0, gpuResult.averageColumnPreview.length);
      for (let index = 0; index < gpuResult.averageColumnPreview.length; index += 1) {
        if (gpuResult.averageColumnPreview[index] !== preview[index]) {
          throw new Error(
            `gpu average-column preview mismatch at column ${index}: ${gpuResult.averageColumnPreview[index]} !== ${preview[index]}`,
          );
        }
      }

      results.push({
        name: gpuResult.name,
        grandTotal: gpuResult.grandTotal,
        averageMs: gpuResult.averageMs,
        medianMs: gpuResult.medianMs,
        minMs: gpuResult.minMs,
        maxMs: gpuResult.maxMs,
        gibPerSecond: gpuResult.gibPerSecond,
      });
    }
  }

  printResults(results);

  const gpuPipeline = compileMatrixReductionGpuPipeline(
    options.matrices,
    options.rows,
    options.cols,
  );
  console.log('');
  if (gpuResults && gpuResults.length > 0) {
    console.log(`GPU device: ${gpuResults[0].deviceName}`);
    for (const gpuResult of gpuResults) {
      if (gpuResult.hostToDeviceCopyAverageMs === null) {
        continue;
      }

      const share = (gpuResult.hostToDeviceCopyAverageMs / gpuResult.averageMs) * 100;
      console.log(
        `GPU mean H2D copy time (${gpuResult.mode}): ${gpuResult.hostToDeviceCopyAverageMs.toFixed(3)} ms (${share.toFixed(2)}% of ${gpuResult.name} avg)`,
      );
    }

    const pageableResult = gpuResults.find((result) => result.mode === 'e2e-pageable');
    const pinnedResult = gpuResults.find((result) => result.mode === 'e2e-pinned');
    if (
      pageableResult &&
      pinnedResult &&
      pageableResult.hostToDeviceCopyAverageMs !== null &&
      pinnedResult.hostToDeviceCopyAverageMs !== null
    ) {
      console.log(
        `GPU H2D pinning speedup: ${(pageableResult.hostToDeviceCopyAverageMs / pinnedResult.hostToDeviceCopyAverageMs).toFixed(2)}x`,
      );
    }
  }
  console.log(`GPU lowering artifact: ${gpuPipeline.stage1KernelIr.split('\n', 1)[0]}`);
  console.log(`GPU lowering artifact: ${gpuPipeline.stage2KernelIr.split('\n', 1)[0]}`);
  console.log(`GPU lowering artifact: ${gpuPipeline.stage3KernelIr.split('\n', 1)[0]}`);
  console.log(`GPU lowering artifact: ${gpuPipeline.stage4KernelIr.split('\n', 1)[0]}`);
  console.log(`GPU PTX size: ${gpuPipeline.ptx.length.toLocaleString()} chars`);
  console.log(`Reference grand total: ${referenceGrandTotal.toFixed(3)}`);
  console.log(
    `Reference averaged columns: ${Array.from(referenceAverageColumnSums.slice(0, PREVIEW_COLUMNS))
      .map((value) => value.toFixed(3))
      .join(', ')}`,
  );
}

function runTask(
  task: BenchmarkTask,
  options: BenchmarkOptions,
  touchedBytes: number,
  referenceGrandTotal: number,
  referenceAverageColumnSums: Float64Array,
): BenchmarkResult {
  for (let iteration = 0; iteration < options.warmup; iteration += 1) {
    task.run();
  }

  const samples: number[] = [];
  let grandTotal = 0;

  for (let iteration = 0; iteration < options.samples; iteration += 1) {
    const startedAt = process.hrtime.bigint();
    grandTotal = task.run();
    const elapsedMs = Number(process.hrtime.bigint() - startedAt) / 1_000_000;
    samples.push(elapsedMs);
  }

  assertSameResults(
    referenceGrandTotal,
    referenceAverageColumnSums,
    grandTotal,
    task.readAverageColumnSums(),
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
    grandTotal,
    averageMs,
    medianMs,
    minMs,
    maxMs,
    gibPerSecond,
  };
}

function assertSameResults(
  expectedGrandTotal: number,
  expectedAverageColumnSums: Float64Array,
  actualGrandTotal: number,
  actualAverageColumnSums: Float64Array,
  label: string,
): void {
  if (actualGrandTotal !== expectedGrandTotal) {
    throw new Error(`${label} grand total mismatch: ${actualGrandTotal} !== ${expectedGrandTotal}`);
  }

  if (actualAverageColumnSums.length !== expectedAverageColumnSums.length) {
    throw new Error(
      `${label} average-column count mismatch: ${actualAverageColumnSums.length} !== ${expectedAverageColumnSums.length}`,
    );
  }

  for (let index = 0; index < expectedAverageColumnSums.length; index += 1) {
    if (actualAverageColumnSums[index] !== expectedAverageColumnSums[index]) {
      throw new Error(
        `${label} average column ${index} mismatch: ${actualAverageColumnSums[index]} !== ${expectedAverageColumnSums[index]}`,
      );
    }
  }
}

function printResults(results: BenchmarkResult[]): void {
  const fastestAverage = Math.min(...results.map((result) => result.averageMs));
  const headers = ['Scenario', 'avg ms', 'med ms', 'min ms', 'max ms', 'GiB/s', 'rel'];
  const rows = results.map((result) => [
    result.name,
    result.averageMs.toFixed(3),
    result.medianMs.toFixed(3),
    result.minMs.toFixed(3),
    result.maxMs.toFixed(3),
    result.gibPerSecond.toFixed(2),
    `${(result.averageMs / fastestAverage).toFixed(2)}x`,
  ]);
  const widths = headers.map((header, index) =>
    Math.max(
      header.length,
      ...rows.map((row) => row[index].length),
    ),
  );
  const rightAligned = new Set([1, 2, 3, 4, 5, 6]);
  const border = `+${widths.map((width) => '-'.repeat(width + 2)).join('+')}+`;

  console.log(border);
  console.log(
    `| ${headers
      .map((header, index) =>
        rightAligned.has(index) ? header.padStart(widths[index]) : header.padEnd(widths[index]),
      )
      .join(' | ')} |`,
  );
  console.log(border);

  for (const row of rows) {
    console.log(
      `| ${row
        .map((cell, index) =>
          rightAligned.has(index) ? cell.padStart(widths[index]) : cell.padEnd(widths[index]),
        )
        .join(' | ')} |`,
    );
  }

  console.log(border);
}

function formatMiB(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
}

function parseArgs(argv: string[]): BenchmarkOptions {
  const options: BenchmarkOptions = {
    matrices: DEFAULT_MATRICES,
    rows: DEFAULT_ROWS,
    cols: DEFAULT_COLS,
    warmup: DEFAULT_WARMUP,
    samples: DEFAULT_SAMPLES,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];

    switch (arg) {
      case '--matrices':
        options.matrices = parsePositiveInt(arg, next);
        index += 1;
        break;
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

  checkedMatrixBatchCells(options.matrices, options.rows, options.cols);

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

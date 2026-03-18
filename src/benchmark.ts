import { spawn, type ChildProcessByStdio } from 'node:child_process';
import { once } from 'node:events';
import { readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import type { Readable, Writable } from 'node:stream';

import { createSharedU32Buffer, mutateSharedU32Buffer } from './index';
import { BufferReader, readResponse, writeRequest } from './ipc-protocol';
import { fillSequence, mutateU32Array } from './mutation';
import { asU32View } from './u32-view';

type BenchmarkOptions = {
  dimension: number;
  passes: number;
  samples: number;
  warmup: number;
  seed: number;
};

type BenchmarkTask = {
  name: string;
  prepare?: () => void | Promise<void>;
  run: () => number | Promise<number>;
  cleanup?: () => void | Promise<void>;
};

type BenchmarkResult = {
  name: string;
  checksum: number;
  averageMs: number;
  medianMs: number;
  minMs: number;
  maxMs: number;
};

type WasmExports = {
  memory: WebAssembly.Memory;
  alloc_u32_buffer(cells: number): number;
  free_u32_buffer(pointer: number, cells: number): void;
  mutate_u32_buffer(pointer: number, cells: number, passes: number): number;
};

class IpcRoundTripClient {
  private readonly reader: BufferReader;

  constructor(
    private readonly name: string,
    private readonly child: ChildProcessByStdio<Writable, Readable, null>,
  ) {
    this.reader = new BufferReader(child.stdout);
  }

  async roundTrip(payload: Buffer, passes: number): Promise<number> {
    await writeRequest(this.child.stdin, { passes, payload });
    const response = await readResponse(this.reader);

    if (response.payload.byteLength !== payload.byteLength) {
      throw new Error(
        `${this.name} returned ${response.payload.byteLength} bytes, expected ${payload.byteLength}`,
      );
    }

    return response.checksum >>> 0;
  }

  async close(): Promise<void> {
    if (this.child.exitCode !== null || this.child.signalCode !== null) {
      if (this.child.exitCode !== 0) {
        throw new Error(
          `${this.name} exited with code ${String(this.child.exitCode)} signal ${String(this.child.signalCode)}`,
        );
      }

      return;
    }

    this.child.stdin.end();
    const [code, signal] = await once(this.child, 'exit');

    if (code !== 0) {
      throw new Error(
        `${this.name} exited with code ${String(code)} signal ${String(signal)}`,
      );
    }
  }
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const cells = checkedCells(options.dimension);
  const byteLength = cells * Uint32Array.BYTES_PER_ELEMENT;

  console.log(
    [
      `Benchmarking ${options.dimension}x${options.dimension} (${cells.toLocaleString()} cells, ${formatMiB(byteLength)})`,
      `passes=${options.passes}`,
      `warmup=${options.warmup}`,
      `samples=${options.samples}`,
    ].join('  '),
  );
  console.log('Direct and WASM timings measure in-process mutation only.');
  console.log('IPC timings measure a full roundtrip: send matrix, mutate in child, return matrix.');
  console.log('');

  const tasks: BenchmarkTask[] = [];
  const sharedCleanup: Array<() => void | Promise<void>> = [];

  try {
    const nativeJsBuffer = createSharedU32Buffer(cells);
    const nativeJsValues = asU32View(nativeJsBuffer, cells);
    tasks.push({
      name: 'native shared / js',
      prepare: () => fillSequence(nativeJsValues, options.seed),
      run: () => mutateU32Array(nativeJsValues, options.passes),
    });

    const nativeRustBuffer = createSharedU32Buffer(cells);
    const nativeRustValues = asU32View(nativeRustBuffer, cells);
    tasks.push({
      name: 'native shared / rust',
      prepare: () => fillSequence(nativeRustValues, options.seed),
      run: () => mutateSharedU32Buffer(nativeRustBuffer, options.passes),
    });

    const wasmJs = await createWasmBuffer(cells);
    sharedCleanup.push(() => wasmJs.dispose());
    tasks.push({
      name: 'wasm memory / js',
      prepare: () => fillSequence(wasmJs.values, options.seed),
      run: () => mutateU32Array(wasmJs.values, options.passes),
    });

    const wasmRust = await createWasmBuffer(cells);
    sharedCleanup.push(() => wasmRust.dispose());
    tasks.push({
      name: 'wasm memory / rust',
      prepare: () => fillSequence(wasmRust.values, options.seed),
      run: () => wasmRust.exports.mutate_u32_buffer(
        wasmRust.pointer,
        cells,
        options.passes,
      ) >>> 0,
    });

    const baseline = createBaselineBuffer(cells, options.seed);

    const nodeIpcClient = new IpcRoundTripClient(
      'node ipc worker',
      spawn(process.execPath, [resolve(__dirname, 'ipc-node-worker.js')], {
        stdio: ['pipe', 'pipe', 'inherit'],
      }),
    );
    tasks.push({
      name: 'ipc roundtrip / js child',
      run: () => nodeIpcClient.roundTrip(baseline, options.passes),
      cleanup: () => nodeIpcClient.close(),
    });

    const rustWorkerPath = resolve(
      __dirname,
      '..',
      'target',
      'release',
      process.platform === 'win32' ? 'ipc_rust_worker.exe' : 'ipc_rust_worker',
    );
    const rustIpcClient = new IpcRoundTripClient(
      'rust ipc worker',
      spawn(rustWorkerPath, [], {
        stdio: ['pipe', 'pipe', 'inherit'],
      }),
    );
    tasks.push({
      name: 'ipc roundtrip / rust child',
      run: () => rustIpcClient.roundTrip(baseline, options.passes),
      cleanup: () => rustIpcClient.close(),
    });

    const results: BenchmarkResult[] = [];

    for (const task of tasks) {
      try {
        results.push(await runTask(task, options));
      } finally {
        await task.cleanup?.();
      }
    }

    const expectedChecksum = results[0]?.checksum;
    for (const result of results) {
      if (result.checksum !== expectedChecksum) {
        throw new Error(
          `Checksum mismatch for ${result.name}: ${result.checksum} !== ${expectedChecksum}`,
        );
      }
    }

    printResults(results);
  } finally {
    for (const cleanup of sharedCleanup.reverse()) {
      await cleanup();
    }
  }
}

function createBaselineBuffer(cells: number, seed: number): Buffer {
  const arrayBuffer = new ArrayBuffer(cells * Uint32Array.BYTES_PER_ELEMENT);
  const values = new Uint32Array(arrayBuffer);
  fillSequence(values, seed);
  return Buffer.from(arrayBuffer);
}

async function createWasmBuffer(cells: number): Promise<{
  exports: WasmExports;
  pointer: number;
  values: Uint32Array;
  dispose: () => void;
}> {
  const wasmPath = join(__dirname, '..', 'wasm', 'pkg', 'matrix_wasm.wasm');
  const wasmBinary = await readFile(wasmPath);
  const { instance } = await WebAssembly.instantiate(wasmBinary, {});
  const exports = instance.exports as unknown as WasmExports;

  if (!(exports.memory instanceof WebAssembly.Memory)) {
    throw new Error('Expected the wasm module to export memory');
  }

  const pointer = exports.alloc_u32_buffer(cells);
  const values = new Uint32Array(exports.memory.buffer, pointer, cells);

  return {
    exports,
    pointer,
    values,
    dispose: () => {
      exports.free_u32_buffer(pointer, cells);
    },
  };
}

async function runTask(
  task: BenchmarkTask,
  options: BenchmarkOptions,
): Promise<BenchmarkResult> {
  for (let iteration = 0; iteration < options.warmup; iteration += 1) {
    await task.prepare?.();
    await Promise.resolve(task.run());
  }

  const samples: number[] = [];
  const checksums = new Set<number>();

  for (let iteration = 0; iteration < options.samples; iteration += 1) {
    await task.prepare?.();
    const startedAt = process.hrtime.bigint();
    const checksum = await Promise.resolve(task.run());
    const elapsedMs = Number(process.hrtime.bigint() - startedAt) / 1_000_000;

    samples.push(elapsedMs);
    checksums.add(checksum >>> 0);
  }

  if (checksums.size !== 1) {
    throw new Error(`${task.name} produced inconsistent checksums across samples`);
  }

  const sorted = [...samples].sort((left, right) => left - right);
  const total = samples.reduce((sum, sample) => sum + sample, 0);
  const averageMs = total / samples.length;
  const medianMs =
    sorted.length % 2 === 0
      ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
      : sorted[Math.floor(sorted.length / 2)];

  return {
    name: task.name,
    checksum: [...checksums][0],
    averageMs,
    medianMs,
    minMs: sorted[0],
    maxMs: sorted[sorted.length - 1],
  };
}

function printResults(results: BenchmarkResult[]): void {
  const fastest = Math.min(...results.map((result) => result.averageMs));

  console.log(
    `${'Scenario'.padEnd(28)} ${'avg ms'.padStart(10)} ${'median'.padStart(10)} ${'min'.padStart(10)} ${'max'.padStart(10)} ${'slower'.padStart(8)}`,
  );

  for (const result of results) {
    const relative = result.averageMs / fastest;
    console.log(
      [
        result.name.padEnd(28),
        result.averageMs.toFixed(2).padStart(10),
        result.medianMs.toFixed(2).padStart(10),
        result.minMs.toFixed(2).padStart(10),
        result.maxMs.toFixed(2).padStart(10),
        `${relative.toFixed(2)}x`.padStart(8),
      ].join(' '),
    );
  }

  console.log('');
  console.log(`checksum: ${results[0]?.checksum ?? 0}`);
}

function parseArgs(argv: string[]): BenchmarkOptions {
  const options: BenchmarkOptions = {
    dimension: 2048,
    passes: 3,
    samples: 5,
    warmup: 1,
    seed: 1,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    const value = argv[index + 1];

    switch (argument) {
      case '--dimension':
        options.dimension = parsePositiveInteger('--dimension', value);
        index += 1;
        break;
      case '--passes':
        options.passes = parsePositiveInteger('--passes', value);
        index += 1;
        break;
      case '--samples':
        options.samples = parsePositiveInteger('--samples', value);
        index += 1;
        break;
      case '--warmup':
        options.warmup = parseNonNegativeInteger('--warmup', value);
        index += 1;
        break;
      case '--seed':
        options.seed = parseNonNegativeInteger('--seed', value);
        index += 1;
        break;
      default:
        throw new Error(`Unknown argument: ${argument}`);
    }
  }

  return options;
}

function parsePositiveInteger(flag: string, value: string | undefined): number {
  if (!value) {
    throw new Error(`Missing value for ${flag}`);
  }

  const parsed = Number.parseInt(value, 10);

  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${flag} must be a positive integer, got ${value}`);
  }

  return parsed;
}

function parseNonNegativeInteger(flag: string, value: string | undefined): number {
  if (!value) {
    throw new Error(`Missing value for ${flag}`);
  }

  const parsed = Number.parseInt(value, 10);

  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${flag} must be a non-negative integer, got ${value}`);
  }

  return parsed;
}

function checkedCells(dimension: number): number {
  const cells = dimension * dimension;

  if (!Number.isSafeInteger(cells)) {
    throw new Error(`Matrix dimension ${dimension} produces an unsafe cell count`);
  }

  return cells;
}

function formatMiB(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

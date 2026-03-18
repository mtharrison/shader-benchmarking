import { execFileSync, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';

export type CudaMatrixBenchmarkOptions = {
  matrices: number;
  rows: number;
  cols: number;
  warmup: number;
  samples: number;
};

export type CudaMatrixBenchmarkResult = {
  name: string;
  deviceName: string;
  averageMs: number;
  medianMs: number;
  minMs: number;
  maxMs: number;
  gibPerSecond: number;
  grandTotal: number;
  averageColumnPreview: number[];
};

export function tryRunCudaMatrixBenchmark(
  options: CudaMatrixBenchmarkOptions,
): CudaMatrixBenchmarkResult | null {
  const rootDir = resolve(__dirname, '..');

  if (!hasCommand('nvidia-smi', ['-L'])) {
    console.error(
      'GPU benchmark unavailable: `nvidia-smi` is not available or no NVIDIA GPU is visible.',
    );
    return null;
  }

  if (!hasCommand('nvcc', ['--version'])) {
    console.error('GPU benchmark unavailable: `nvcc` is not available in PATH.');
    return null;
  }

  const runnerPath = buildCudaRunner(rootDir);
  if (!runnerPath) {
    return null;
  }

  try {
    const output = execFileSync(
      runnerPath,
      [
        '--matrices',
        String(options.matrices),
        '--rows',
        String(options.rows),
        '--cols',
        String(options.cols),
        '--warmup',
        String(options.warmup),
        '--samples',
        String(options.samples),
      ],
      {
        cwd: rootDir,
        encoding: 'utf8',
      },
    );

    return JSON.parse(output) as CudaMatrixBenchmarkResult;
  } catch (error) {
    console.error(
      `GPU benchmark failed while executing the CUDA runner.\n${formatExecError(error)}`,
    );
    return null;
  }
}

function hasCommand(command: string, args: string[]): boolean {
  const result = spawnSync(command, args, { encoding: 'utf8' });
  return !result.error && result.status === 0;
}

function buildCudaRunner(rootDir: string): string | null {
  const sourcePath = join(rootDir, 'cuda', 'matrix_reduction_runner.cu');
  const outputDir = join(rootDir, 'target', 'cuda');
  const outputPath = join(
    outputDir,
    process.platform === 'win32' ? 'matrix_reduction_runner.exe' : 'matrix_reduction_runner',
  );

  mkdirSync(outputDir, { recursive: true });

  const needsRebuild =
    !existsSync(outputPath) ||
    statSync(outputPath).mtimeMs < statSync(sourcePath).mtimeMs;

  if (!needsRebuild) {
    return outputPath;
  }

  try {
    execFileSync(
      'nvcc',
      ['-O3', '-std=c++17', sourcePath, '-o', outputPath],
      {
        cwd: rootDir,
        encoding: 'utf8',
      },
    );

    return outputPath;
  } catch (error) {
    console.error(
      `GPU benchmark unavailable: failed to compile the CUDA runner with \`nvcc\`.\n${formatExecError(error)}`,
    );
    return null;
  }
}

function formatExecError(error: unknown): string {
  if (
    typeof error === 'object' &&
    error !== null &&
    'stderr' in error &&
    'stdout' in error
  ) {
    const stderr = bufferLikeToString((error as { stderr: unknown }).stderr);
    const stdout = bufferLikeToString((error as { stdout: unknown }).stdout);

    return [stderr, stdout].filter(Boolean).join('\n');
  }

  if (error instanceof Error) {
    return error.message;
  }

  return String(error);
}

function bufferLikeToString(value: unknown): string {
  if (typeof value === 'string') {
    return value.trim();
  }

  if (Buffer.isBuffer(value)) {
    return value.toString('utf8').trim();
  }

  return '';
}

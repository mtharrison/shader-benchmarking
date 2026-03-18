import { copyFileSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';
import { platform } from 'node:os';

const rootDir = resolve(dirname(new URL(import.meta.url).pathname), '..');
const targetDir = join(rootDir, 'target', 'release');
const outputDir = join(rootDir, 'native');
const outputFile = join(outputDir, 'index.node');

const libraryName = (() => {
  switch (platform()) {
    case 'darwin':
      return 'libsharing_memory.dylib';
    case 'linux':
      return 'libsharing_memory.so';
    case 'win32':
      return 'sharing_memory.dll';
    default:
      throw new Error(`Unsupported platform: ${platform()}`);
  }
})();

execFileSync('cargo', ['build', '--release'], {
  cwd: rootDir,
  stdio: 'inherit',
});

mkdirSync(outputDir, { recursive: true });
copyFileSync(join(targetDir, libraryName), outputFile);

console.log(`Native module written to ${outputFile}`);

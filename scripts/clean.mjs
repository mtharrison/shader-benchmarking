import { rmSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

const rootDir = resolve(dirname(new URL(import.meta.url).pathname), '..');

rmSync(join(rootDir, 'dist'), { force: true, recursive: true });
rmSync(join(rootDir, 'native', 'index.node'), { force: true });
rmSync(join(rootDir, 'wasm', 'pkg'), { force: true, recursive: true });

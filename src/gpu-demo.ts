import {
  compileGpuMap2Pipeline,
  compileSampleGpuMap2,
  sampleGpuMap2SourceCode,
} from './index';

function printSection(title: string, body: string): void {
  console.log(`\n=== ${title} ===`);
  console.log(body);
}

const sampleSource = sampleGpuMap2SourceCode();
const sample = compileSampleGpuMap2();
const alternate = compileGpuMap2Pipeline(
  'let out = gpu.map2(lhs, rhs, (x, y) => x * y);',
);

printSection('Input Source', sampleSource);
printSection('JS AST', sample.jsAst);
printSection('Kernel IR', sample.kernelIr);
printSection('PTX', sample.ptx);
printSection('Host Launch Sketch', sample.hostLaunch);
printSection('Notes', sample.notes.map((note) => `- ${note}`).join('\n'));

printSection('Alternate Kernel Name', alternate.kernelIr.split('\n', 1)[0]);

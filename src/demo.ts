import {
  asMatrixView,
  createSharedMatrix,
  printMatrix6x6,
  printNativeMatrix6x6,
} from './index';

const sharedBuffer = createSharedMatrix();
const matrix = asMatrixView(sharedBuffer);

console.log('TypeScript printer on the initial shared matrix:');
printMatrix6x6(matrix);

matrix[0] = 9001;
matrix[7] = 1234;
matrix[35] = 4242;

console.log('\nTypeScript printer after mutating the shared buffer:');
printMatrix6x6(matrix);

console.log('\nRust printer reading the same shared buffer:');
printNativeMatrix6x6(sharedBuffer);

import { stdin, stdout } from 'node:process';

import { BufferReader, readRequest, writeResponse } from './ipc-protocol';
import { mutateU32Array } from './mutation';
import { asU32View } from './u32-view';

async function main(): Promise<void> {
  const reader = new BufferReader(stdin);

  while (true) {
    const request = await readRequest(reader);

    if (!request) {
      break;
    }

    const checksum = mutateU32Array(asU32View(request.payload), request.passes);

    await writeResponse(stdout, { checksum, payload: request.payload });
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

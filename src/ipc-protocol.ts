import { once } from 'node:events';
import type { Readable, Writable } from 'node:stream';

const HEADER_BYTES = 8;

export type IpcRequest = {
  passes: number;
  payload: Buffer;
};

export type IpcResponse = {
  checksum: number;
  payload: Buffer;
};

export class BufferReader {
  private readonly chunks: Buffer[] = [];
  private readonly waiters: Array<() => void> = [];
  private bufferedBytes = 0;
  private ended = false;
  private error: Error | null = null;

  constructor(stream: Readable) {
    stream.on('data', (chunk: Buffer | string) => {
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      this.chunks.push(buffer);
      this.bufferedBytes += buffer.length;
      this.wake();
    });

    stream.on('end', () => {
      this.ended = true;
      this.wake();
    });

    stream.on('error', (error: Error) => {
      this.error = error;
      this.wake();
    });
  }

  async readExactOrNull(size: number): Promise<Buffer | null> {
    while (this.bufferedBytes < size) {
      if (this.error) {
        throw this.error;
      }

      if (this.ended) {
        if (this.bufferedBytes === 0) {
          return null;
        }

        throw new Error(`Expected ${size} bytes before EOF, got ${this.bufferedBytes}`);
      }

      await this.waitForMore();
    }

    return this.consume(size);
  }

  async readExact(size: number): Promise<Buffer> {
    const chunk = await this.readExactOrNull(size);

    if (!chunk) {
      throw new Error(`Expected ${size} bytes before EOF`);
    }

    return chunk;
  }

  private async waitForMore(): Promise<void> {
    await new Promise<void>((resolve) => {
      this.waiters.push(resolve);
    });
  }

  private wake(): void {
    while (this.waiters.length > 0) {
      const waiter = this.waiters.shift();
      waiter?.();
    }
  }

  private consume(size: number): Buffer {
    this.bufferedBytes -= size;

    if (this.chunks[0]?.length === size) {
      return this.chunks.shift() as Buffer;
    }

    const result = Buffer.allocUnsafe(size);
    let offset = 0;

    while (offset < size) {
      const chunk = this.chunks[0] as Buffer;
      const remaining = size - offset;

      if (chunk.length <= remaining) {
        chunk.copy(result, offset);
        offset += chunk.length;
        this.chunks.shift();
        continue;
      }

      chunk.copy(result, offset, 0, remaining);
      this.chunks[0] = chunk.subarray(remaining);
      offset += remaining;
    }

    return result;
  }
}

export async function readRequest(reader: BufferReader): Promise<IpcRequest | null> {
  const header = await reader.readExactOrNull(HEADER_BYTES);

  if (!header) {
    return null;
  }

  const passes = header.readUInt32LE(0);
  const byteLength = header.readUInt32LE(4);
  const payload = await reader.readExact(byteLength);

  return { passes, payload };
}

export async function readResponse(reader: BufferReader): Promise<IpcResponse> {
  const header = await reader.readExact(HEADER_BYTES);
  const checksum = header.readUInt32LE(0);
  const byteLength = header.readUInt32LE(4);
  const payload = await reader.readExact(byteLength);

  return { checksum, payload };
}

export async function writeRequest(
  stream: Writable,
  request: IpcRequest,
): Promise<void> {
  const header = Buffer.allocUnsafe(HEADER_BYTES);
  header.writeUInt32LE(request.passes >>> 0, 0);
  header.writeUInt32LE(request.payload.byteLength >>> 0, 4);

  await writeAll(stream, header);
  await writeAll(stream, request.payload);
}

export async function writeResponse(
  stream: Writable,
  response: IpcResponse,
): Promise<void> {
  const header = Buffer.allocUnsafe(HEADER_BYTES);
  header.writeUInt32LE(response.checksum >>> 0, 0);
  header.writeUInt32LE(response.payload.byteLength >>> 0, 4);

  await writeAll(stream, header);
  await writeAll(stream, response.payload);
}

async function writeAll(stream: Writable, chunk: Buffer): Promise<void> {
  if (stream.write(chunk)) {
    return;
  }

  await once(stream, 'drain');
}

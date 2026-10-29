import { describe, expect, it } from 'vitest';
import { crc32, zipParts } from '../src/core/zip.js';

const concat = (parts) => {
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
};

function readZip(bytes) {
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const end = bytes.length - 22;
  expect(v.getUint32(end, true)).toBe(0x06054b50);
  const count = v.getUint16(end + 10, true);
  let cd = v.getUint32(end + 16, true);
  const entries = [];
  for (let i = 0; i < count; i++) {
    expect(v.getUint32(cd, true)).toBe(0x02014b50);
    const crc = v.getUint32(cd + 16, true);
    const size = v.getUint32(cd + 24, true);
    const nameLen = v.getUint16(cd + 28, true);
    const local = v.getUint32(cd + 42, true);
    const name = new TextDecoder().decode(bytes.subarray(cd + 46, cd + 46 + nameLen));
    expect(v.getUint32(local, true)).toBe(0x04034b50);
    const start = local + 30 + v.getUint16(local + 26, true);
    entries.push({ name, crc, data: bytes.subarray(start, start + size) });
    cd += 46 + nameLen;
  }
  return entries;
}

describe('crc32', () => {
  it('matches the standard check value', () => {
    expect(crc32(new TextEncoder().encode('123456789'))).toBe(0xcbf43926);
    expect(crc32(new Uint8Array())).toBe(0);
  });
});

describe('zipParts', () => {
  it('writes a readable archive with unique UTF-8 names', () => {
    const a = new Uint8Array([1, 2, 3, 4, 5]);
    const b = new TextEncoder().encode('hello ünïcode');
    const bytes = concat(zipParts([
      { name: 'photo.png', data: a },
      { name: 'photo.png', data: b },
      { name: 'café.jpg', data: new Uint8Array() },
    ]));
    const entries = readZip(bytes);
    expect(entries.map((e) => e.name)).toEqual(['photo.png', 'photo (2).png', 'café.jpg']);
    expect(Array.from(entries[0].data)).toEqual([1, 2, 3, 4, 5]);
    expect(new TextDecoder().decode(entries[1].data)).toBe('hello ünïcode');
    expect(entries[1].crc).toBe(crc32(b));
    expect(entries[2].data.length).toBe(0);
  });

  it('writes a valid empty archive', () => {
    const bytes = concat(zipParts([]));
    expect(bytes.length).toBe(22);
    expect(readZip(bytes)).toEqual([]);
  });
});

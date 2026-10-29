import { describe, expect, it } from 'vitest';
import { estimateJpegQuality, readExif, sniffFormat, ZIGZAG } from '../src/core/exif.js';
import { rng } from './helpers.js';

const STD = [
  16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
  14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
  49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

function tiff() {
  const buf = new ArrayBuffer(200);
  const v = new DataView(buf);
  const le = true;
  v.setUint8(0, 0x49);
  v.setUint8(1, 0x49);
  v.setUint16(2, 42, le);
  v.setUint32(4, 8, le);
  const entry = (o, tag, type, count, value, inlineShort = false) => {
    v.setUint16(o, tag, le);
    v.setUint16(o + 2, type, le);
    v.setUint32(o + 4, count, le);
    if (inlineShort) v.setUint16(o + 8, value, le);
    else v.setUint32(o + 8, value, le);
  };
  v.setUint16(8, 4, le);
  entry(10, 0x010f, 2, 6, 100); // Make -> "Canon"
  entry(22, 0x0112, 3, 1, 6, true); // Orientation 6
  entry(34, 0x8769, 4, 1, 120); // Exif IFD
  entry(46, 0x8825, 4, 1, 170); // GPS IFD
  v.setUint32(58, 0, le);
  'Canon\0'.split('').forEach((c, i) => v.setUint8(100 + i, c.charCodeAt(0)));
  v.setUint16(120, 2, le);
  entry(122, 0x8827, 3, 1, 400, true); // ISO
  entry(134, 0x829d, 5, 1, 150); // FNumber
  v.setUint32(146, 0, le);
  v.setUint32(150, 28, le);
  v.setUint32(154, 10, le);
  v.setUint16(170, 0, le);
  return new Uint8Array(buf);
}

function dqt(quality) {
  const scale = quality < 50 ? 5000 / quality : 200 - quality * 2;
  const natural = STD.map((q) => Math.min(255, Math.max(1, Math.floor((q * scale + 50) / 100))));
  return ZIGZAG.map((idx) => natural[idx]);
}

function jpeg({ withExif = true, quality = 75 } = {}) {
  const parts = [0xff, 0xd8];
  if (withExif) {
    const t = tiff();
    const len = 2 + 6 + t.length;
    parts.push(0xff, 0xe1, len >> 8, len & 255, 0x45, 0x78, 0x69, 0x66, 0, 0, ...t);
  }
  parts.push(0xff, 0xdb, 0, 67, 0x00, ...dqt(quality));
  parts.push(0xff, 0xd9);
  return new Uint8Array(parts);
}

describe('sniffFormat', () => {
  it('recognises common containers', () => {
    expect(sniffFormat(jpeg())).toBe('JPEG');
    expect(sniffFormat(new Uint8Array([0x89, 0x50, 0x4e, 0x47, 13, 10, 26, 10]))).toBe('PNG');
    const webp = new TextEncoder().encode('RIFF\0\0\0\0WEBPVP8 ');
    expect(sniffFormat(webp)).toBe('WebP');
    const heic = new Uint8Array([0, 0, 0, 24, ...new TextEncoder().encode('ftypheic')]);
    expect(sniffFormat(heic)).toBe('HEIC');
    expect(sniffFormat(new Uint8Array([1, 2, 3]))).toBe('Unknown');
  });
});

describe('readExif', () => {
  it('reads camera tags from a JPEG', () => {
    const exif = readExif(jpeg());
    expect(exif.make).toBe('Canon');
    expect(exif.orientation).toBe(6);
    expect(exif.iso).toBe(400);
    expect(exif.fNumber).toBeCloseTo(2.8, 5);
    expect(exif.hasGps).toBe(true);
  });

  it('returns null without EXIF', () => {
    expect(readExif(jpeg({ withExif: false }))).toBeNull();
  });

  it('never throws on garbage or truncated files', () => {
    const rand = rng(11);
    for (let i = 0; i < 200; i++) {
      const bytes = new Uint8Array(Math.floor(rand() * 300)).map(() => Math.floor(rand() * 256));
      if (i % 3 === 0 && bytes.length > 3) bytes.set([0xff, 0xd8, 0xff]);
      expect(() => readExif(bytes)).not.toThrow();
      expect(() => estimateJpegQuality(bytes)).not.toThrow();
    }
    const full = jpeg();
    for (let cut = 0; cut < full.length; cut += 7) {
      expect(() => readExif(full.subarray(0, cut))).not.toThrow();
    }
  });
});

describe('estimateJpegQuality', () => {
  it('recovers the IJG quality setting', () => {
    for (const q of [30, 50, 75, 90]) {
      expect(Math.abs(estimateJpegQuality(jpeg({ withExif: false, quality: q })) - q)).toBeLessThanOrEqual(2);
    }
  });

  it('returns null for non-JPEG data', () => {
    expect(estimateJpegQuality(new Uint8Array([0x89, 0x50, 0x4e, 0x47]))).toBeNull();
  });
});

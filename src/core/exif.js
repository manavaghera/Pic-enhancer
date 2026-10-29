// Minimal, defensive metadata reader: container sniffing, EXIF (JPEG APP1,
// PNG eXIf, WebP EXIF chunk) and JPEG quality estimation from DQT tables.
// Every offset is bounds-checked; malformed files yield null, never throw.

const IFD0_TAGS = { 0x010f: 'make', 0x0110: 'model', 0x0112: 'orientation', 0x0131: 'software', 0x0132: 'dateTime' };
const EXIF_TAGS = {
  0x829a: 'exposureTime',
  0x829d: 'fNumber',
  0x8827: 'iso',
  0x9003: 'dateTimeOriginal',
  0x9209: 'flash',
  0x920a: 'focalLength',
  0xa001: 'colorSpace',
  0xa405: 'focalLength35mm',
  0xa434: 'lensModel',
};
const TYPE_SIZES = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8 };

export function sniffFormat(bytes) {
  const b = bytes;
  const ascii = (o, len) => String.fromCharCode(...b.subarray(o, o + len));
  if (b.length >= 3 && b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return 'JPEG';
  if (b.length >= 8 && b[0] === 0x89 && ascii(1, 3) === 'PNG') return 'PNG';
  if (b.length >= 12 && ascii(0, 4) === 'RIFF' && ascii(8, 4) === 'WEBP') return 'WebP';
  if (b.length >= 6 && ascii(0, 3) === 'GIF') return 'GIF';
  if (b.length >= 2 && ascii(0, 2) === 'BM') return 'BMP';
  if (b.length >= 12 && ascii(4, 4) === 'ftyp') {
    const brand = ascii(8, 4);
    if (brand === 'avif' || brand === 'avis') return 'AVIF';
    if (/^(heic|heix|hevc|mif1|msf1)$/.test(brand)) return 'HEIC';
  }
  return 'Unknown';
}

function tiffFromJpeg(b) {
  let o = 2;
  while (o + 4 <= b.length) {
    if (b[o] !== 0xff) return null;
    const marker = b[o + 1];
    if (marker === 0xd9 || marker === 0xda) return null;
    const len = (b[o + 2] << 8) | b[o + 3];
    if (len < 2) return null;
    if (marker === 0xe1 && o + 10 <= b.length &&
        String.fromCharCode(b[o + 4], b[o + 5], b[o + 6], b[o + 7]) === 'Exif' && b[o + 8] === 0 && b[o + 9] === 0) {
      return b.subarray(o + 10, Math.min(b.length, o + 2 + len));
    }
    o += 2 + len;
  }
  return null;
}

function tiffFromPng(b) {
  let o = 8;
  while (o + 12 <= b.length) {
    const len = ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3]) >>> 0;
    const type = String.fromCharCode(b[o + 4], b[o + 5], b[o + 6], b[o + 7]);
    if (type === 'eXIf') return b.subarray(o + 8, Math.min(b.length, o + 8 + len));
    if (type === 'IEND') return null;
    o += 12 + len;
  }
  return null;
}

function tiffFromWebp(b) {
  let o = 12;
  while (o + 8 <= b.length) {
    const type = String.fromCharCode(b[o], b[o + 1], b[o + 2], b[o + 3]);
    const len = (b[o + 4] | (b[o + 5] << 8) | (b[o + 6] << 16) | (b[o + 7] << 24)) >>> 0;
    if (type === 'EXIF') {
      let start = o + 8;
      if (String.fromCharCode(b[start], b[start + 1], b[start + 2], b[start + 3]) === 'Exif') start += 6;
      return b.subarray(start, Math.min(b.length, o + 8 + len));
    }
    o += 8 + len + (len & 1);
  }
  return null;
}

function readIfd(view, offset, little, names, out) {
  if (offset < 8 || offset + 2 > view.byteLength) return { exifPtr: 0, gpsPtr: 0 };
  const count = view.getUint16(offset, little);
  let exifPtr = 0, gpsPtr = 0;
  for (let i = 0; i < Math.min(count, 512); i++) {
    const e = offset + 2 + i * 12;
    if (e + 12 > view.byteLength) break;
    const tag = view.getUint16(e, little);
    const type = view.getUint16(e + 2, little);
    const num = view.getUint32(e + 4, little);
    if (tag === 0x8769) { exifPtr = view.getUint32(e + 8, little); continue; }
    if (tag === 0x8825) { gpsPtr = view.getUint32(e + 8, little); continue; }
    const name = names[tag];
    const size = TYPE_SIZES[type];
    if (!name || !size || num === 0 || num > 4096) continue;
    const valOff = size * num <= 4 ? e + 8 : view.getUint32(e + 8, little);
    if (valOff + size * num > view.byteLength) continue;
    out[name] = readValue(view, valOff, type, num, little);
  }
  return { exifPtr, gpsPtr };
}

function readValue(view, o, type, num, little) {
  switch (type) {
    case 2: {
      let s = '';
      for (let i = 0; i < num; i++) {
        const c = view.getUint8(o + i);
        if (c === 0) break;
        s += String.fromCharCode(c);
      }
      return s.trim();
    }
    case 3: return view.getUint16(o, little);
    case 4: return view.getUint32(o, little);
    case 9: return view.getInt32(o, little);
    case 5:
    case 10: {
      const read = type === 5 ? 'getUint32' : 'getInt32';
      const n = view[read](o, little), d = view[read](o + 4, little);
      return d === 0 ? null : n / d;
    }
    default: return view.getUint8(o);
  }
}

export function parseTiff(tiff) {
  if (!tiff || tiff.length < 8) return null;
  const view = new DataView(tiff.buffer, tiff.byteOffset, tiff.byteLength);
  const order = view.getUint16(0, false);
  if (order !== 0x4949 && order !== 0x4d4d) return null;
  const little = order === 0x4949;
  if (view.getUint16(2, little) !== 42) return null;
  const out = {};
  const { exifPtr, gpsPtr } = readIfd(view, view.getUint32(4, little), little, IFD0_TAGS, out);
  if (exifPtr) readIfd(view, exifPtr, little, EXIF_TAGS, out);
  out.hasGps = gpsPtr >= 8 && gpsPtr < view.byteLength;
  return out;
}

export function readExif(bytes) {
  try {
    const format = sniffFormat(bytes);
    const tiff = format === 'JPEG' ? tiffFromJpeg(bytes) : format === 'PNG' ? tiffFromPng(bytes) : format === 'WebP' ? tiffFromWebp(bytes) : null;
    const exif = parseTiff(tiff);
    if (!exif) return null;
    return Object.keys(exif).length > 1 || exif.hasGps ? exif : null;
  } catch {
    return null;
  }
}

// Standard IJG luminance table (natural order) and the zig-zag scan order.
const STD_LUMA = [
  16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
  14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
  49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];
export const ZIGZAG = [
  0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
  35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

// Inverse of the IJG quality scaling applied to the first luminance table.
export function estimateJpegQuality(bytes) {
  try {
    if (sniffFormat(bytes) !== 'JPEG') return null;
    let o = 2;
    while (o + 4 <= bytes.length) {
      if (bytes[o] !== 0xff) return null;
      const marker = bytes[o + 1];
      if (marker === 0xda || marker === 0xd9) return null;
      const len = (bytes[o + 2] << 8) | bytes[o + 3];
      if (len < 2) return null;
      if (marker === 0xdb) {
        let p = o + 4;
        const end = Math.min(bytes.length, o + 2 + len);
        while (p < end) {
          const pq = bytes[p] >> 4, tq = bytes[p] & 15;
          const size = pq ? 128 : 64;
          if (p + 1 + size > end) return null;
          if (tq === 0) {
            let ratio = 0;
            for (let k = 0; k < 64; k++) {
              const q = pq ? (bytes[p + 1 + 2 * k] << 8) | bytes[p + 2 + 2 * k] : bytes[p + 1 + k];
              ratio += (q * 100) / STD_LUMA[ZIGZAG[k]];
            }
            const scale = ratio / 64;
            const quality = scale <= 100 ? (200 - scale) / 2 : 5000 / scale;
            return Math.max(1, Math.min(100, Math.round(quality)));
          }
          p += 1 + size;
        }
      }
      o += 2 + len;
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Minimal perceptual-image helpers for the reference-asset invariant test.
 *
 * The reference-set guard must detect when a curated reference PNG is the SAME
 * PHOTO as a study stimulus — including the case the byte-SHA guard misses: the
 * clean seed photo re-encoded by the pipeline (pixel-identical, byte-different).
 * That requires comparing decoded pixels, not bytes.
 *
 * We avoid a heavyweight image dependency: every PNG in this study (refs,
 * committed stimuli, and run-dir seed origins) is uniformly 8-bit, color-type 2
 * (RGB), non-interlaced — see the curation tooling. So a small, exactly-scoped
 * RGB PNG decoder built on Node's zlib is sufficient and keeps the test
 * dependency-free. It THROWS on any PNG outside that profile rather than
 * silently mis-decoding.
 *
 * Two perceptual signals, mirroring experiments/HS-01/tools/build_references.py:
 *   - pixelMae  : mean abs grayscale difference at a fixed size (~0 => same photo)
 *   - dhash     : difference hash; small Hamming distance => same scene
 */

import fs from "fs";
import zlib from "zlib";

export const CMP_SIZE = 64; // grayscale-resize edge for pixel-MAE
export const DHASH_SIZE = 16; // dHash edge -> 16*16 = 256-bit hash

/** Decoded grayscale image at native resolution. */
interface Gray {
  width: number;
  height: number;
  /** Row-major luma in [0,255], length width*height. */
  data: Float64Array;
}

const PNG_SIG = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

function paeth(a: number, b: number, c: number): number {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

/** Decode an 8-bit RGB (color type 2), non-interlaced PNG to grayscale luma. */
export function decodeGray(filePath: string): Gray {
  const buf = fs.readFileSync(filePath);
  if (!buf.subarray(0, 8).equals(PNG_SIG)) {
    throw new Error(`${filePath}: not a PNG`);
  }
  let off = 8;
  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = 0;
  let interlace = 0;
  const idat: Buffer[] = [];

  while (off < buf.length) {
    const len = buf.readUInt32BE(off);
    const type = buf.toString("ascii", off + 4, off + 8);
    const dataStart = off + 8;
    if (type === "IHDR") {
      width = buf.readUInt32BE(dataStart);
      height = buf.readUInt32BE(dataStart + 4);
      bitDepth = buf[dataStart + 8];
      colorType = buf[dataStart + 9];
      interlace = buf[dataStart + 12];
    } else if (type === "IDAT") {
      idat.push(buf.subarray(dataStart, dataStart + len));
    } else if (type === "IEND") {
      break;
    }
    off = dataStart + len + 4; // skip data + CRC
  }

  if (bitDepth !== 8 || colorType !== 2 || interlace !== 0) {
    throw new Error(
      `${filePath}: unsupported PNG profile ` +
        `(bitDepth=${bitDepth} colorType=${colorType} interlace=${interlace}); ` +
        `decoder handles only 8-bit RGB non-interlaced`
    );
  }

  const raw = zlib.inflateSync(Buffer.concat(idat));
  const channels = 3;
  const stride = width * channels;
  // Unfilter scanlines in place into a recon buffer.
  const recon = Buffer.alloc(height * stride);
  let rp = 0; // pointer into raw
  for (let y = 0; y < height; y++) {
    const filter = raw[rp++];
    const rowStart = y * stride;
    const prevStart = rowStart - stride;
    for (let x = 0; x < stride; x++) {
      const rawVal = raw[rp++];
      const a = x >= channels ? recon[rowStart + x - channels] : 0;
      const b = y > 0 ? recon[prevStart + x] : 0;
      const c = y > 0 && x >= channels ? recon[prevStart + x - channels] : 0;
      let val: number;
      switch (filter) {
        case 0:
          val = rawVal;
          break;
        case 1:
          val = rawVal + a;
          break;
        case 2:
          val = rawVal + b;
          break;
        case 3:
          val = rawVal + ((a + b) >> 1);
          break;
        case 4:
          val = rawVal + paeth(a, b, c);
          break;
        default:
          throw new Error(`${filePath}: bad PNG filter type ${filter}`);
      }
      recon[rowStart + x] = val & 0xff;
    }
  }

  // Rec.601 luma.
  const data = new Float64Array(width * height);
  for (let i = 0, px = 0; i < recon.length; i += channels, px++) {
    data[px] = 0.299 * recon[i] + 0.587 * recon[i + 1] + 0.114 * recon[i + 2];
  }
  return { width, height, data };
}

/** Bilinear-ish box resample of a grayscale image to size x size. */
function resizeGray(src: Gray, size: number): Float64Array {
  const out = new Float64Array(size * size);
  for (let oy = 0; oy < size; oy++) {
    const syf = ((oy + 0.5) * src.height) / size - 0.5;
    const sy0 = Math.max(0, Math.min(src.height - 1, Math.floor(syf)));
    const sy1 = Math.min(src.height - 1, sy0 + 1);
    const wy = Math.max(0, Math.min(1, syf - sy0));
    for (let ox = 0; ox < size; ox++) {
      const sxf = ((ox + 0.5) * src.width) / size - 0.5;
      const sx0 = Math.max(0, Math.min(src.width - 1, Math.floor(sxf)));
      const sx1 = Math.min(src.width - 1, sx0 + 1);
      const wx = Math.max(0, Math.min(1, sxf - sx0));
      const v00 = src.data[sy0 * src.width + sx0];
      const v01 = src.data[sy0 * src.width + sx1];
      const v10 = src.data[sy1 * src.width + sx0];
      const v11 = src.data[sy1 * src.width + sx1];
      const top = v00 + (v01 - v00) * wx;
      const bot = v10 + (v11 - v10) * wx;
      out[oy * size + ox] = top + (bot - top) * wy;
    }
  }
  return out;
}

const _grayCache = new Map<string, Gray>();
function grayOf(p: string): Gray {
  let g = _grayCache.get(p);
  if (!g) {
    g = decodeGray(p);
    _grayCache.set(p, g);
  }
  return g;
}

const _maeResizeCache = new Map<string, Float64Array>();
function maeResized(p: string): Float64Array {
  let r = _maeResizeCache.get(p);
  if (!r) {
    r = resizeGray(grayOf(p), CMP_SIZE);
    _maeResizeCache.set(p, r);
  }
  return r;
}

/** Mean absolute grayscale pixel difference at CMP_SIZE; ~0 => same photo. */
export function pixelMae(a: string, b: string): number {
  const ga = maeResized(a);
  const gb = maeResized(b);
  let s = 0;
  for (let i = 0; i < ga.length; i++) s += Math.abs(ga[i] - gb[i]);
  return s / ga.length;
}

const _dhashCache = new Map<string, Uint8Array>();
/** Difference hash (DHASH_SIZE^2 bits) as a 0/1 byte array. */
export function dhash(p: string): Uint8Array {
  let h = _dhashCache.get(p);
  if (h) return h;
  const r = resizeGray(grayOf(p), DHASH_SIZE + 1); // (size+1) x (size+1) box
  // Use the first DHASH_SIZE rows; compare horizontally within each row.
  const bits = new Uint8Array(DHASH_SIZE * DHASH_SIZE);
  const w = DHASH_SIZE + 1;
  for (let y = 0; y < DHASH_SIZE; y++) {
    for (let x = 0; x < DHASH_SIZE; x++) {
      bits[y * DHASH_SIZE + x] = r[y * w + x + 1] > r[y * w + x] ? 1 : 0;
    }
  }
  _dhashCache.set(p, bits);
  return h ?? bits;
}

export function hamming(a: Uint8Array, b: Uint8Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) d++;
  return d;
}

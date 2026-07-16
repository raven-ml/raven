/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* JavaScript stubs for extended bigarray types.
   Extends the standard js_of_ocaml bigarray implementation.

   Supported extended types:
   - bfloat16: Brain floating-point (16-bit)
   - bool: Boolean values (8-bit)
   - int4_signed/unsigned: 4-bit integers (packed 2 per byte)
   - float8_e4m3/e5m2: 8-bit floating-point formats
   - uint32/uint64: Unsigned 32/64-bit integers

   The implementation extends the standard Ml_Bigarray class with
   Ml_Nx_buffer to handle get/set/fill operations for these types.

   The numeric conversions mirror nx_buffer_stubs.h bit for bit (round to
   nearest, ties to even; NaN and saturation handling) so that programs
   compute the same values under js_of_ocaml as natively, except for NaN
   payload bits, which JavaScript canonicalizes. */

//Provides: caml_unpackBfloat16
function caml_unpackBfloat16(bits) {
  /* bfloat16 is the upper 16 bits of a float32 */
  var buffer = new ArrayBuffer(4);
  var view = new DataView(buffer);
  view.setUint32(0, (bits & 0xffff) << 16, false);
  return view.getFloat32(0, false);
}

//Provides: caml_packBfloat16
function caml_packBfloat16(num) {
  var buffer = new ArrayBuffer(4);
  var view = new DataView(buffer);
  view.setFloat32(0, num, false);
  var bits = view.getUint32(0, false);
  /* NaN first: the rounding bias below could carry a small NaN significand
     into the exponent and turn it into inf. */
  if ((bits & 0x7fffffff) > 0x7f800000) {
    return ((bits >>> 16) | 0x0040) & 0xffff;
  }
  /* Round to nearest even */
  var rounding_bias = ((bits >>> 16) & 1) + 0x7fff;
  return ((bits + rounding_bias) >>> 16) & 0xffff;
}

//Provides: caml_packFp8_e4m3
function caml_packFp8_e4m3(num) {
  /* OCP "fn" variant: no infinities, S.1111.111 is NaN, exponent 15 is
     otherwise normal up to the max finite 448, subnormals scale by 2^-6.
     Finite overflow and infinities convert to NaN, matching the ml_dtypes
     and PyTorch e4m3fn casts. */
  if (num !== num) return 0x7f;
  if (num === Infinity) return 0x7f;
  if (num === -Infinity) return 0xff;

  var buffer = new ArrayBuffer(4);
  var view = new DataView(buffer);
  view.setFloat32(0, num, false);
  var bits = view.getUint32(0, false);
  var sign = ((bits >>> 31) << 7) & 0xff;
  var exp = ((bits >>> 23) & 0xff) - 127;

  if (exp >= -6) {
    /* Normal range: round the 23-bit significand to 3 bits, ties to even. */
    var sig = bits & 0x7fffff;
    var q = sig >>> 20;
    var rem = sig & 0xfffff;
    if (rem > 0x80000 || (rem === 0x80000 && (q & 1))) q++;
    /* A rounding carry propagates into the exponent field. */
    var out = ((exp + 7) << 3) + q;
    if (out >= 0x7f) return sign | 0x7f; /* Overflow past 448 to NaN */
    return sign | out;
  }

  /* Subnormal or zero: denormalize to the 2^-6 scale, keeping all
     shifted-out bits for the rounding decision. */
  var sig2 = (bits & 0x7fffff) | 0x800000;
  var shift = 20 + (-6 - exp);
  if (shift > 24) return sign; /* Below half the min subnormal 2^-9 */
  var q2 = sig2 >>> shift;
  var rem2 = sig2 & ((1 << shift) - 1);
  var half = 1 << (shift - 1);
  if (rem2 > half || (rem2 === half && (q2 & 1))) q2++;
  /* q2 == 8 after rounding is the min normal; the bit pattern lines up. */
  return sign | q2;
}

//Provides: caml_unpackFp8_e4m3
function caml_unpackFp8_e4m3(byte) {
  var negative = (byte & 0x80) !== 0;
  var exp = (byte >>> 3) & 0xf;
  var mant = byte & 0x7;
  /* No infinities: exponent 15 is normal except S.1111.111. */
  if (exp === 0xf && mant === 0x7) return NaN;
  var v;
  if (exp === 0) {
    v = mant * Math.pow(2, -9); /* Subnormal: mant/8 * 2^-6 */
  } else {
    v = (1 + mant / 8) * Math.pow(2, exp - 7);
  }
  return negative ? -v : v;
}

//Provides: caml_packFp8_e5m2
function caml_packFp8_e5m2(num) {
  /* IEEE-like: has infinities and subnormals. Finite overflow rounds to
     infinity. */
  if (num !== num) return 0x7f;
  if (num === Infinity) return 0x7c;
  if (num === -Infinity) return 0xfc;

  var buffer = new ArrayBuffer(4);
  var view = new DataView(buffer);
  view.setFloat32(0, num, false);
  var bits = view.getUint32(0, false);
  var sign = ((bits >>> 31) << 7) & 0xff;
  var exp = ((bits >>> 23) & 0xff) - 127;

  if (exp >= -14) {
    /* Normal range: round the 23-bit significand to 2 bits, ties to even. */
    var sig = bits & 0x7fffff;
    var q = sig >>> 21;
    var rem = sig & 0x1fffff;
    if (rem > 0x100000 || (rem === 0x100000 && (q & 1))) q++;
    /* A rounding carry propagates into the exponent field. */
    var out = ((exp + 15) << 2) + q;
    if (out >= 0x7c) return sign | 0x7c; /* Overflow to Inf */
    return sign | out;
  }

  /* Subnormal or zero: denormalize to the 2^-14 scale, keeping all
     shifted-out bits for the rounding decision. */
  var sig2 = (bits & 0x7fffff) | 0x800000;
  var shift = 21 + (-14 - exp);
  if (shift > 24) return sign; /* Below half the min subnormal 2^-16 */
  var q2 = sig2 >>> shift;
  var rem2 = sig2 & ((1 << shift) - 1);
  var half = 1 << (shift - 1);
  if (rem2 > half || (rem2 === half && (q2 & 1))) q2++;
  /* q2 == 4 after rounding is the min normal; the bit pattern lines up. */
  return sign | q2;
}

//Provides: caml_unpackFp8_e5m2
function caml_unpackFp8_e5m2(byte) {
  var negative = (byte & 0x80) !== 0;
  var exp = (byte >>> 2) & 0x1f;
  var mant = byte & 0x3;
  if (exp === 0x1f) {
    if (mant !== 0) return NaN;
    return negative ? -Infinity : Infinity;
  }
  var v;
  if (exp === 0) {
    v = mant * Math.pow(2, -16); /* Subnormal: mant/4 * 2^-14 */
  } else {
    v = (1 + mant / 4) * Math.pow(2, exp - 15);
  }
  return negative ? -v : v;
}

/* Extended kind enumeration continuing the js_of_ocaml runtime kinds
   (0=Float32 .. 13=Float16), matching the C NX_BA_* enumeration. */
var NX_BA_BFLOAT16 = 14;
var NX_BA_BOOL = 15;
var NX_BA_INT4 = 16;
var NX_BA_UINT4 = 17;
var NX_BA_FP8_E4M3 = 18;
var NX_BA_FP8_E5M2 = 19;
var NX_BA_UINT32 = 20;
var NX_BA_UINT64 = 21;

//Provides: caml_nx_buffer_size_per_element
//Requires: caml_ba_get_size_per_element
function caml_nx_buffer_size_per_element(kind) {
  /* Typed-array slots per logical element, like
     caml_ba_get_size_per_element. */
  if (kind < 14) {
    return caml_ba_get_size_per_element(kind);
  }
  switch (kind) {
    case 21: /* NX_BA_UINT64: lo/hi pairs, like the standard Int64 */
      return 2;
    default:
      return 1;
  }
}

//Provides: caml_nx_buffer_create_data
//Requires: caml_ba_create_buffer
//Requires: caml_invalid_argument
function caml_nx_buffer_create_data(kind, size) {
  /* Handle standard types */
  if (kind < 14) {
    return caml_ba_create_buffer(kind, size);
  }

  switch (kind) {
    case 14: /* NX_BA_BFLOAT16 */
      return new Uint16Array(size);
    case 15: /* NX_BA_BOOL */
    case 18: /* NX_BA_FP8_E4M3 */
    case 19: /* NX_BA_FP8_E5M2 */
      return new Uint8Array(size);
    case 16: /* NX_BA_INT4 */
    case 17: /* NX_BA_UINT4 */
      /* Pack 2 values per byte */
      return new Uint8Array(Math.ceil(size / 2));
    case 20: /* NX_BA_UINT32 */
      return new Uint32Array(size);
    case 21: /* NX_BA_UINT64: lo/hi pairs, like the standard Int64 */
      return new Int32Array(size * 2);
    default:
      caml_invalid_argument("Bigarray.create: unsupported extended kind");
  }
}

//Provides: Ml_Nx_buffer
//Requires: Ml_Bigarray, caml_invalid_argument
//Requires: caml_unpackBfloat16, caml_packBfloat16
//Requires: caml_unpackFp8_e4m3, caml_packFp8_e4m3
//Requires: caml_unpackFp8_e5m2, caml_packFp8_e5m2
//Requires: caml_int64_create_lo_hi, caml_int64_lo32, caml_int64_hi32
class Ml_Nx_buffer extends Ml_Bigarray {
  get(ofs) {
    /* Handle standard types */
    if (this.kind < 14) {
      return super.get(ofs);
    }

    /* Handle extended types */
    switch (this.kind) {
      case 14: /* NX_BA_BFLOAT16 */
        return caml_unpackBfloat16(this.data[ofs]);
      case 15: /* NX_BA_BOOL */
        return this.data[ofs] ? 1 : 0;
      case 16: { /* NX_BA_INT4 */
        var byte = this.data[Math.floor(ofs / 2)];
        var val;
        if (ofs % 2 === 0) {
          val = byte & 0x0f;
        } else {
          val = (byte >> 4) & 0x0f;
        }
        /* Sign extend */
        if (val & 0x08) val -= 16;
        return val;
      }
      case 17: { /* NX_BA_UINT4 */
        var byte = this.data[Math.floor(ofs / 2)];
        if (ofs % 2 === 0) {
          return byte & 0x0f;
        } else {
          return (byte >> 4) & 0x0f;
        }
      }
      case 18: /* NX_BA_FP8_E4M3 */
        return caml_unpackFp8_e4m3(this.data[ofs]);
      case 19: /* NX_BA_FP8_E5M2 */
        return caml_unpackFp8_e5m2(this.data[ofs]);
      case 20: /* NX_BA_UINT32 */
        return this.data[ofs] | 0;
      case 21: /* NX_BA_UINT64 */
        return caml_int64_create_lo_hi(
          this.data[ofs * 2 + 0],
          this.data[ofs * 2 + 1]
        );
      default:
        return this.data[ofs];
    }
  }

  set(ofs, v) {
    /* Handle standard types */
    if (this.kind < 14) {
      return super.set(ofs, v);
    }

    /* Handle extended types */
    switch (this.kind) {
      case 14: /* NX_BA_BFLOAT16 */
        this.data[ofs] = caml_packBfloat16(v);
        break;
      case 15: /* NX_BA_BOOL */
        this.data[ofs] = v ? 1 : 0;
        break;
      case 16: { /* NX_BA_INT4: clamp to [-8, 7], like the C stub */
        if (v > 7) v = 7;
        if (v < -8) v = -8;
        var byte_idx = Math.floor(ofs / 2);
        var byte = this.data[byte_idx];
        if (ofs % 2 === 0) {
          this.data[byte_idx] = (byte & 0xf0) | (v & 0x0f);
        } else {
          this.data[byte_idx] = (byte & 0x0f) | ((v & 0x0f) << 4);
        }
        break;
      }
      case 17: { /* NX_BA_UINT4: clamp to [0, 15], like the C stub */
        if (v > 15) v = 15;
        if (v < 0) v = 0;
        var byte_idx = Math.floor(ofs / 2);
        var byte = this.data[byte_idx];
        if (ofs % 2 === 0) {
          this.data[byte_idx] = (byte & 0xf0) | (v & 0x0f);
        } else {
          this.data[byte_idx] = (byte & 0x0f) | ((v & 0x0f) << 4);
        }
        break;
      }
      case 18: /* NX_BA_FP8_E4M3 */
        this.data[ofs] = caml_packFp8_e4m3(v);
        break;
      case 19: /* NX_BA_FP8_E5M2 */
        this.data[ofs] = caml_packFp8_e5m2(v);
        break;
      case 20: /* NX_BA_UINT32 */
        this.data[ofs] = v >>> 0;
        break;
      case 21: /* NX_BA_UINT64 */
        this.data[ofs * 2 + 0] = caml_int64_lo32(v);
        this.data[ofs * 2 + 1] = caml_int64_hi32(v);
        break;
      default:
        this.data[ofs] = v;
        break;
    }
    return 0;
  }

  fill(v) {
    /* Handle standard types */
    if (this.kind < 14) {
      return super.fill(v);
    }

    /* Handle extended types */
    switch (this.kind) {
      case 14: /* NX_BA_BFLOAT16 */
        this.data.fill(caml_packBfloat16(v));
        break;
      case 15: /* NX_BA_BOOL */
        this.data.fill(v ? 1 : 0);
        break;
      case 16: /* NX_BA_INT4: clamp to [-8, 7], like the C stub */
        if (v > 7) v = 7;
        if (v < -8) v = -8;
        this.data.fill(((v & 0x0f) << 4) | (v & 0x0f));
        break;
      case 17: /* NX_BA_UINT4: clamp to [0, 15], like the C stub */
        if (v > 15) v = 15;
        if (v < 0) v = 0;
        this.data.fill(((v & 0x0f) << 4) | (v & 0x0f));
        break;
      case 18: /* NX_BA_FP8_E4M3 */
        this.data.fill(caml_packFp8_e4m3(v));
        break;
      case 19: /* NX_BA_FP8_E5M2 */
        this.data.fill(caml_packFp8_e5m2(v));
        break;
      case 20: /* NX_BA_UINT32 */
        this.data.fill(v >>> 0);
        break;
      case 21: { /* NX_BA_UINT64 */
        var lo = caml_int64_lo32(v);
        var hi = caml_int64_hi32(v);
        for (var i = 0; i < this.data.length; i += 2) {
          this.data[i] = lo;
          this.data[i + 1] = hi;
        }
        break;
      }
      default:
        this.data.fill(v);
        break;
    }
  }
}

//Provides: caml_nx_buffer_create_unsafe
//Requires: Ml_Nx_buffer, Ml_Bigarray_c_1_1, Ml_Bigarray
//Requires: caml_ba_get_size, caml_nx_buffer_size_per_element
//Requires: caml_invalid_argument
function caml_nx_buffer_create_unsafe(kind, layout, dims, data) {
  var num_elts = caml_ba_get_size(dims);

  /* Int4/uint4 pack two elements per byte */
  if (kind === 16 || kind === 17) {
    if (Math.ceil(num_elts / 2) !== data.length) {
      caml_invalid_argument("length doesn't match dims (int4/uint4)");
    }
  } else if (
    num_elts * caml_nx_buffer_size_per_element(kind) !==
    data.length
  ) {
    caml_invalid_argument("length doesn't match dims");
  }

  /* Use extended class for extended types */
  if (kind >= 14) {
    return new Ml_Nx_buffer(kind, layout, dims, data);
  }

  /* Use standard classes for standard types */
  if (
    layout === 0 && /* c_layout */
    dims.length === 1 && /* Array1 */
    caml_nx_buffer_size_per_element(kind) === 1 &&
    kind !== 13 /* float16 */
  ) {
    return new Ml_Bigarray_c_1_1(kind, layout, dims, data);
  }
  return new Ml_Bigarray(kind, layout, dims, data);
}

//Provides: caml_nx_buffer_create_internal
//Requires: caml_js_from_array
//Requires: caml_ba_get_size, caml_nx_buffer_create_unsafe
//Requires: caml_nx_buffer_create_data
function caml_nx_buffer_create_internal(kind, layout, dims_ml) {
  var dims = caml_js_from_array(dims_ml);
  var data = caml_nx_buffer_create_data(kind, caml_ba_get_size(dims));
  return caml_nx_buffer_create_unsafe(kind, layout, dims, data);
}

/* Creation functions for each extended type */
//Provides: caml_nx_buffer_create_bfloat16
//Requires: caml_nx_buffer_create_internal
function caml_nx_buffer_create_bfloat16(layout, dims) {
  return caml_nx_buffer_create_internal(14, layout, dims);
}

//Provides: caml_nx_buffer_create_bool
//Requires: caml_nx_buffer_create_internal
function caml_nx_buffer_create_bool(layout, dims) {
  return caml_nx_buffer_create_internal(15, layout, dims);
}

//Provides: caml_nx_buffer_create_int4_signed
//Requires: caml_nx_buffer_create_internal
function caml_nx_buffer_create_int4_signed(layout, dims) {
  return caml_nx_buffer_create_internal(16, layout, dims);
}

//Provides: caml_nx_buffer_create_int4_unsigned
//Requires: caml_nx_buffer_create_internal
function caml_nx_buffer_create_int4_unsigned(layout, dims) {
  return caml_nx_buffer_create_internal(17, layout, dims);
}

//Provides: caml_nx_buffer_create_float8_e4m3
//Requires: caml_nx_buffer_create_internal
function caml_nx_buffer_create_float8_e4m3(layout, dims) {
  return caml_nx_buffer_create_internal(18, layout, dims);
}

//Provides: caml_nx_buffer_create_float8_e5m2
//Requires: caml_nx_buffer_create_internal
function caml_nx_buffer_create_float8_e5m2(layout, dims) {
  return caml_nx_buffer_create_internal(19, layout, dims);
}

//Provides: caml_nx_buffer_create_uint32
//Requires: caml_nx_buffer_create_internal
function caml_nx_buffer_create_uint32(layout, dims) {
  return caml_nx_buffer_create_internal(20, layout, dims);
}

//Provides: caml_nx_buffer_create_uint64
//Requires: caml_nx_buffer_create_internal
function caml_nx_buffer_create_uint64(layout, dims) {
  return caml_nx_buffer_create_internal(21, layout, dims);
}

//Provides: caml_nx_buffer_get
//Requires: caml_js_from_array, caml_ba_get_generic
function caml_nx_buffer_get(ba, i) {
  /* If it's an extended bigarray, use its get method */
  if (ba.kind >= 14) {
    var ofs = ba.offset(caml_js_from_array(i));
    return ba.get(ofs);
  }
  /* Otherwise use standard implementation */
  return caml_ba_get_generic(ba, i);
}

//Provides: caml_nx_buffer_set
//Requires: caml_js_from_array, caml_ba_set_generic
function caml_nx_buffer_set(ba, i, v) {
  /* If it's an extended bigarray, use its set method */
  if (ba.kind >= 14) {
    ba.set(ba.offset(caml_js_from_array(i)), v);
    return 0;
  }
  /* Otherwise use standard implementation */
  return caml_ba_set_generic(ba, i, v);
}

//Provides: caml_nx_buffer_unsafe_get
//Requires: caml_ba_get_1
function caml_nx_buffer_unsafe_get(ba, i) {
  if (ba.kind >= 14) {
    return ba.get(i);
  }
  return caml_ba_get_1(ba, i);
}

//Provides: caml_nx_buffer_unsafe_set
//Requires: caml_ba_set_1
function caml_nx_buffer_unsafe_set(ba, i, v) {
  if (ba.kind >= 14) {
    ba.set(i, v);
    return 0;
  }
  return caml_ba_set_1(ba, i, v);
}

//Provides: caml_nx_buffer_kind
//Requires: caml_failwith
function caml_nx_buffer_kind(ba) {
  /* Map runtime bigarray kind to the GADT constructor index. Pinned to the
     declaration order of [Nx_buffer.kind] (19 constructors) and mirrored by
     the C stub. The js_of_ocaml runtime numbers kinds like the C runtime:
     0=Float32, 1=Float64, 2=Int8s, 3=Uint8, 4=Int16s, 5=Uint16, 6=Int32,
     7=Int64, 8=Int, 9=Nativeint, 10=Complex32, 11=Complex64, 12=Char,
     13=Float16, then our extended kinds 14-21. */
  switch (ba.kind) {
    case 13: return 0;  /* Float16 */
    case 0: return 1;   /* Float32 */
    case 1: return 2;   /* Float64 */
    case 14: return 3;  /* BFloat16 */
    case 18: return 4;  /* Float8_e4m3 */
    case 19: return 5;  /* Float8_e5m2 */
    case 16: return 6;  /* Int4 */
    case 17: return 7;  /* UInt4 */
    case 2: return 8;   /* Int8 */
    case 3: return 9;   /* UInt8 */
    case 4: return 10;  /* Int16 */
    case 5: return 11;  /* UInt16 */
    case 6: return 12;  /* Int32 */
    case 20: return 13; /* UInt32 */
    case 7: return 14;  /* Int64 */
    case 21: return 15; /* UInt64 */
    case 10: return 16; /* Complex64 */
    case 11: return 17; /* Complex128 */
    case 15: return 18; /* Bool */
    default:
      caml_failwith("Unknown bigarray kind: " + ba.kind);
  }
}

//Provides: caml_nx_buffer_blit
//Requires: caml_ba_blit, caml_invalid_argument
function caml_nx_buffer_blit(src, dst) {
  if (src.kind >= 14 && dst.kind >= 14 && src.kind === dst.kind) {
    /* For extended types, raw data copy */
    if (src.data.length !== dst.data.length) {
      caml_invalid_argument("Nx_buffer.blit: arrays have different dimensions");
    }
    dst.data.set(src.data);
    return 0;
  }
  return caml_ba_blit(src, dst);
}

//Provides: caml_nx_buffer_fill
//Requires: caml_ba_fill
function caml_nx_buffer_fill(ba, v) {
  if (ba.kind >= 14) {
    ba.fill(v);
    return 0;
  }
  return caml_ba_fill(ba, v);
}

//Provides: caml_nx_buffer_blit_from_bytes
//Requires: caml_bytes_unsafe_get
function caml_nx_buffer_blit_from_bytes(bytes, src_off, dst, dst_off, len) {
  var dst_data = new Uint8Array(dst.data.buffer, dst.data.byteOffset);
  for (var i = 0; i < len; i++) {
    dst_data[dst_off + i] = caml_bytes_unsafe_get(bytes, src_off + i);
  }
  return 0;
}

//Provides: caml_nx_buffer_blit_to_bytes
//Requires: caml_bytes_unsafe_set
function caml_nx_buffer_blit_to_bytes(src, src_off, bytes, dst_off, len) {
  var src_data = new Uint8Array(src.data.buffer, src.data.byteOffset);
  for (var i = 0; i < len; i++) {
    caml_bytes_unsafe_set(bytes, dst_off + i, src_data[src_off + i]);
  }
  return 0;
}

//Provides: caml_nx_buffer_data_ptr
//Requires: caml_failwith
function caml_nx_buffer_data_ptr(ba) {
  caml_failwith("Nx_buffer.unsafe_data_ptr: not supported on JavaScript");
}

/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

// JavaScript stubs for extended bigarray types
// Extends the standard js_of_ocaml bigarray implementation
//
// This file provides JavaScript support for the following extended types:
// - bfloat16: Brain floating-point (16-bit)
// - bool: Boolean values (8-bit)
// - int4_signed/unsigned: 4-bit integers (packed 2 per byte)
// - float8_e4m3/e5m2: 8-bit floating-point formats
// - uint32/uint64: Unsigned 32/64-bit integers
//
// The implementation extends the standard Ml_Bigarray class with
// Ml_Nx_buffer to handle get/set/fill operations for these types.

//Provides: caml_unpackBfloat16
function caml_unpackBfloat16(bytes) {
  // bfloat16 is the upper 16 bits of a float32
  var buffer = new ArrayBuffer(4);
  var view = new DataView(buffer);
  view.setUint16(2, bytes, false); // big-endian, upper 16 bits
  view.setUint16(0, 0, false); // lower 16 bits are zero
  return view.getFloat32(0, false);
}

//Provides: caml_packBfloat16
function caml_packBfloat16(num) {
  // Convert to float32 and take upper 16 bits
  var buffer = new ArrayBuffer(4);
  var view = new DataView(buffer);
  view.setFloat32(0, num, false);
  return view.getUint16(2, false);
}

//Provides: caml_unpackFp8_e4m3
function caml_unpackFp8_e4m3(byte) {
  var sign = (byte >> 7) & 0x1;
  var exp = (byte >> 3) & 0xf;
  var mantissa = byte & 0x7;
  
  if (exp === 0 && mantissa === 0) return sign ? -0.0 : 0.0;
  
  // Convert to float32 format
  exp = exp - 7 + 127; // Remove E4M3 bias, add float32 bias
  var bits = (sign << 31) | (exp << 23) | (mantissa << 20);
  
  var buffer = new ArrayBuffer(4);
  var view = new DataView(buffer);
  view.setUint32(0, bits, false);
  return view.getFloat32(0, false);
}

//Provides: caml_packFp8_e4m3
function caml_packFp8_e4m3(num) {
  var buffer = new ArrayBuffer(4);
  var view = new DataView(buffer);
  view.setFloat32(0, num, false);
  var bits = view.getUint32(0, false);
  
  var sign = (bits >> 31) & 0x1;
  var exp = ((bits >> 23) & 0xff) - 127; // Extract and unbias exponent
  var mantissa = (bits >> 20) & 0x7; // Take top 3 bits of mantissa
  
  // Clamp exponent to E4M3 range [-6, 8] with bias 7
  exp = exp + 7; // Apply E4M3 bias
  if (exp <= 0) exp = 0;
  if (exp >= 15) exp = 15;
  
  return (sign << 7) | ((exp & 0xf) << 3) | (mantissa & 0x7);
}

//Provides: caml_unpackFp8_e5m2
function caml_unpackFp8_e5m2(byte) {
  var sign = (byte >> 7) & 0x1;
  var exp = (byte >> 2) & 0x1f;
  var mantissa = byte & 0x3;
  
  if (exp === 0 && mantissa === 0) return sign ? -0.0 : 0.0;
  
  // Convert to float32 format
  exp = exp - 15 + 127; // Remove E5M2 bias, add float32 bias
  var bits = (sign << 31) | (exp << 23) | (mantissa << 21);
  
  var buffer = new ArrayBuffer(4);
  var view = new DataView(buffer);
  view.setUint32(0, bits, false);
  return view.getFloat32(0, false);
}

//Provides: caml_packFp8_e5m2
function caml_packFp8_e5m2(num) {
  var buffer = new ArrayBuffer(4);
  var view = new DataView(buffer);
  view.setFloat32(0, num, false);
  var bits = view.getUint32(0, false);
  
  var sign = (bits >> 31) & 0x1;
  var exp = ((bits >> 23) & 0xff) - 127; // Extract and unbias exponent
  var mantissa = (bits >> 21) & 0x3; // Take top 2 bits of mantissa
  
  // Clamp exponent to E5M2 range [-14, 15] with bias 15
  exp = exp + 15; // Apply E5M2 bias
  if (exp <= 0) exp = 0;
  if (exp >= 31) exp = 31;
  
  return (sign << 7) | ((exp & 0x1f) << 2) | (mantissa & 0x3);
}

// Extended kind enumeration matching our C implementation
var NX_BA_BFLOAT16 = 14;
var NX_BA_BOOL = 15;
var NX_BA_INT4 = 16;
var NX_BA_UINT4 = 17;
var NX_BA_FP8_E4M3 = 18;
var NX_BA_FP8_E5M2 = 19;
var NX_BA_UINT32 = 20;
var NX_BA_UINT64 = 21;

//Provides: caml_nx_ba_get_size_per_element
//Requires: caml_ba_get_size_per_element
function caml_nx_ba_get_size_per_element(kind) {
  // Handle standard types first
  if (kind < 14) {
    return caml_ba_get_size_per_element(kind);
  }
  
  // Handle extended types
  switch (kind) {
    case 14: // NX_BA_BFLOAT16
      return 2;
    case 15: // NX_BA_BOOL
      return 1;
    case 16: // NX_BA_INT4
    case 17: // NX_BA_UINT4
      return 1; // Packed 2 per byte
    case 18: // NX_BA_FP8_E4M3
    case 19: // NX_BA_FP8_E5M2
      return 1;
    case 20: // NX_BA_UINT32
      return 4;
    case 21: // NX_BA_UINT64
      return 8;
    default:
      return 1;
  }
}

//Provides: caml_nx_ba_create_buffer
//Requires: caml_ba_create_buffer, caml_nx_ba_get_size_per_element
//Requires: caml_invalid_argument
function caml_nx_ba_create_buffer(kind, size) {
  // Handle standard types
  if (kind < 14) {
    return caml_ba_create_buffer(kind, size);
  }
  
  // For extended types, use appropriate typed arrays
  var view;
  switch (kind) {
    case 14: // NX_BA_BFLOAT16
      view = Uint16Array;
      break;
    case 15: // NX_BA_BOOL
      view = Uint8Array;
      break;
    case 16: // NX_BA_INT4
    case 17: // NX_BA_UINT4
      // Pack 2 values per byte
      view = Uint8Array;
      size = Math.ceil(size / 2);
      break;
    case 18: // NX_BA_FP8_E4M3
    case 19: // NX_BA_FP8_E5M2
      view = Uint8Array;
      break;
    case 20: // NX_BA_UINT32
      view = Uint32Array;
      break;
    case 21: // NX_BA_UINT64
      if (typeof BigUint64Array === "undefined") {
        caml_invalid_argument("Bigarray.create: uint64 not supported");
      }
      view = BigUint64Array;
      break;
    default:
      caml_invalid_argument("Bigarray.create: unsupported extended kind");
  }
  
  return new view(size);
}

//Provides: Ml_Nx_buffer
//Requires: Ml_Bigarray, caml_invalid_argument
//Requires: caml_unpackBfloat16, caml_packBfloat16
//Requires: caml_unpackFp8_e4m3, caml_packFp8_e4m3
//Requires: caml_unpackFp8_e5m2, caml_packFp8_e5m2
class Ml_Nx_buffer extends Ml_Bigarray {
  get(ofs) {
    // Handle standard types
    if (this.kind < 14) {
      return super.get(ofs);
    }
    
    // Handle extended types
    switch (this.kind) {
      case 14: // NX_BA_BFLOAT16
        return caml_unpackBfloat16(this.data[ofs]);
      case 15: // NX_BA_BOOL
        return this.data[ofs] ? 1 : 0;
      case 16: { // NX_BA_INT4
        var byte = this.data[Math.floor(ofs / 2)];
        var val;
        if (ofs % 2 === 0) {
          val = (byte & 0x0f); // Lower 4 bits
          // Sign extend
          if (val & 0x08) val |= 0xfffffff0;
        } else {
          val = (byte >> 4) & 0x0f; // Upper 4 bits
          // Sign extend
          if (val & 0x08) val |= 0xfffffff0;
        }
        return val;
      }
      case 17: { // NX_BA_UINT4
        var byte = this.data[Math.floor(ofs / 2)];
        if (ofs % 2 === 0) {
          return byte & 0x0f; // Lower 4 bits
        } else {
          return (byte >> 4) & 0x0f; // Upper 4 bits
        }
      }
      case 18: // NX_BA_FP8_E4M3
        return caml_unpackFp8_e4m3(this.data[ofs]);
      case 19: // NX_BA_FP8_E5M2
        return caml_unpackFp8_e5m2(this.data[ofs]);
      case 20: // NX_BA_UINT32
        return this.data[ofs] | 0;
      case 21: // NX_BA_UINT64
        return BigInt.asIntN(64, this.data[ofs]);
      default:
        return this.data[ofs];
    }
  }
  
  set(ofs, v) {
    // Handle standard types
    if (this.kind < 14) {
      return super.set(ofs, v);
    }
    
    // Handle extended types
    switch (this.kind) {
      case 14: // NX_BA_BFLOAT16
        this.data[ofs] = caml_packBfloat16(v);
        break;
      case 15: // NX_BA_BOOL
        this.data[ofs] = v ? 1 : 0;
        break;
      case 16: { // NX_BA_INT4
        if (v < -8 || v > 7) {
          caml_invalid_argument("Bigarray.set: int4 value out of range [-8, 7]");
        }
        var byte_idx = Math.floor(ofs / 2);
        var byte = this.data[byte_idx];
        if (ofs % 2 === 0) {
          this.data[byte_idx] = (byte & 0xf0) | (v & 0x0f); // Set lower 4 bits
        } else {
          this.data[byte_idx] = (byte & 0x0f) | ((v & 0x0f) << 4); // Set upper 4 bits
        }
        break;
      }
      case 17: { // NX_BA_UINT4
        if (v < 0 || v > 15) {
          caml_invalid_argument("Bigarray.set: uint4 value out of range [0, 15]");
        }
        var byte_idx = Math.floor(ofs / 2);
        var byte = this.data[byte_idx];
        if (ofs % 2 === 0) {
          this.data[byte_idx] = (byte & 0xf0) | (v & 0x0f); // Set lower 4 bits
        } else {
          this.data[byte_idx] = (byte & 0x0f) | ((v & 0x0f) << 4); // Set upper 4 bits
        }
        break;
      }
      case 18: // NX_BA_FP8_E4M3
        this.data[ofs] = caml_packFp8_e4m3(v);
        break;
      case 19: // NX_BA_FP8_E5M2
        this.data[ofs] = caml_packFp8_e5m2(v);
        break;
      case 20: // NX_BA_UINT32
        this.data[ofs] = v >>> 0;
        break;
      case 21: // NX_BA_UINT64
        this.data[ofs] = BigInt.asUintN(64, v);
        break;
      default:
        this.data[ofs] = v;
        break;
    }
    return 0;
  }
  
  fill(v) {
    // Handle standard types
    if (this.kind < 14) {
      return super.fill(v);
    }
    
    // Handle extended types
    switch (this.kind) {
      case 14: // NX_BA_BFLOAT16
        this.data.fill(caml_packBfloat16(v));
        break;
      case 15: // NX_BA_BOOL
        this.data.fill(v ? 1 : 0);
        break;
      case 16: // NX_BA_INT4
      case 17: // NX_BA_UINT4
        // For int4/uint4, we need to pack 2 values per byte
        var packed = (v & 0x0f) | ((v & 0x0f) << 4);
        this.data.fill(packed);
        break;
      case 18: // NX_BA_FP8_E4M3
        this.data.fill(caml_packFp8_e4m3(v));
        break;
      case 19: // NX_BA_FP8_E5M2
        this.data.fill(caml_packFp8_e5m2(v));
        break;
      case 20: // NX_BA_UINT32
        this.data.fill(v >>> 0);
        break;
      case 21: // NX_BA_UINT64
        this.data.fill(BigInt.asUintN(64, v));
        break;
      default:
        this.data.fill(v);
        break;
    }
  }
}

//Provides: caml_nx_ba_create_unsafe
//Requires: Ml_Nx_buffer, Ml_Bigarray_c_1_1, Ml_Bigarray
//Requires: caml_ba_get_size, caml_nx_ba_get_size_per_element
//Requires: caml_invalid_argument
function caml_nx_ba_create_unsafe(kind, layout, dims, data) {
  var size_per_element = caml_nx_ba_get_size_per_element(kind);
  
  // For int4/uint4, adjust size calculation
  if (kind === 16 || kind === 17) {
    var num_elts = caml_ba_get_size(dims);
    if (Math.ceil(num_elts / 2) !== data.length) {
      caml_invalid_argument("length doesn't match dims (int4/uint4)");
    }
  } else if (caml_ba_get_size(dims) * size_per_element !== data.length) {
    caml_invalid_argument("length doesn't match dims");
  }
  
  // Use extended class for extended types
  if (kind >= 14) {
    return new Ml_Nx_buffer(kind, layout, dims, data);
  }
  
  // Use standard classes for standard types
  if (
    layout === 0 && // c_layout
    dims.length === 1 && // Array1
    size_per_element === 1 &&
    kind !== 13 // float16
  ) {
    return new Ml_Bigarray_c_1_1(kind, layout, dims, data);
  }
  return new Ml_Bigarray(kind, layout, dims, data);
}

//Provides: caml_nx_ba_create
//Requires: caml_js_from_array
//Requires: caml_ba_get_size, caml_nx_ba_create_unsafe
//Requires: caml_nx_ba_create_buffer
function caml_nx_ba_create(kind, layout, dims_ml) {
  var dims = caml_js_from_array(dims_ml);
  var data = caml_nx_ba_create_buffer(kind, caml_ba_get_size(dims));
  return caml_nx_ba_create_unsafe(kind, layout, dims, data);
}

// Creation functions for each extended type
//Provides: caml_nx_ba_create_bfloat16
//Requires: caml_nx_ba_create
function caml_nx_ba_create_bfloat16(layout, dims) {
  return caml_nx_ba_create(14, layout, dims);
}

//Provides: caml_nx_ba_create_bool
//Requires: caml_nx_ba_create
function caml_nx_ba_create_bool(layout, dims) {
  return caml_nx_ba_create(15, layout, dims);
}

//Provides: caml_nx_ba_create_int4_signed
//Requires: caml_nx_ba_create
function caml_nx_ba_create_int4_signed(layout, dims) {
  return caml_nx_ba_create(16, layout, dims);
}

//Provides: caml_nx_ba_create_int4_unsigned
//Requires: caml_nx_ba_create
function caml_nx_ba_create_int4_unsigned(layout, dims) {
  return caml_nx_ba_create(17, layout, dims);
}

//Provides: caml_nx_ba_create_float8_e4m3
//Requires: caml_nx_ba_create
function caml_nx_ba_create_float8_e4m3(layout, dims) {
  return caml_nx_ba_create(18, layout, dims);
}

//Provides: caml_nx_ba_create_float8_e5m2
//Requires: caml_nx_ba_create
function caml_nx_ba_create_float8_e5m2(layout, dims) {
  return caml_nx_ba_create(19, layout, dims);
}

//Provides: caml_nx_ba_create_uint32
//Requires: caml_nx_ba_create
function caml_nx_ba_create_uint32(layout, dims) {
  return caml_nx_ba_create(20, layout, dims);
}

//Provides: caml_nx_ba_create_uint64
//Requires: caml_nx_ba_create
function caml_nx_ba_create_uint64(layout, dims) {
  return caml_nx_ba_create(21, layout, dims);
}

//Provides: caml_nx_ba_get_generic
//Requires: caml_js_from_array, caml_ba_get_generic
function caml_nx_ba_get_generic(ba, i) {
  // If it's an extended bigarray, use its get method
  if (ba.kind >= 14) {
    var ofs = ba.offset(caml_js_from_array(i));
    return ba.get(ofs);
  }
  // Otherwise use standard implementation
  return caml_ba_get_generic(ba, i);
}

//Provides: caml_nx_ba_set_generic
//Requires: caml_js_from_array, caml_ba_set_generic
function caml_nx_ba_set_generic(ba, i, v) {
  // If it's an extended bigarray, use its set method
  if (ba.kind >= 14) {
    ba.set(ba.offset(caml_js_from_array(i)), v);
    return 0;
  }
  // Otherwise use standard implementation
  return caml_ba_set_generic(ba, i, v);
}

//Provides: caml_nx_ba_kind
//Requires: Ml_Nx_buffer
function caml_nx_ba_kind(ba) {
  // Map bigarray kind to our extended kind enum values
  // These must match the OCaml type constructor order
  switch (ba.kind) {
    case 1: return 0;  // Float32
    case 0: return 1;  // Float64
    case 2: return 2;  // Int8_signed
    case 3: return 3;  // Int8_unsigned
    case 4: return 4;  // Int16_signed
    case 5: return 5;  // Int16_unsigned
    case 8: return 6;  // Int32
    case 9: return 7;  // Int64
    case 10: return 8; // Int
    case 11: return 9; // Nativeint
    case 6: return 10; // Complex32
    case 7: return 11; // Complex64
    case 12: return 12; // Char
    case 13: return 13; // Float16
    // Extended types
    case 14: return 14; // Bfloat16
    case 15: return 15; // Bool
    case 16: return 16; // Int4_signed
    case 17: return 17; // Int4_unsigned
    case 18: return 18; // Float8_e4m3
    case 19: return 19; // Float8_e5m2
    case 20: return 20; // Uint32
    case 21: return 21; // Uint64
    default:
      throw new Error("Unknown bigarray kind: " + ba.kind);
  }
}

# SIMD Optimization Audit for nx-oxcaml Kernels

## Summary Table: Kernel SIMD Status by Data Type

| Kernel                    | Float32  | Float64  | Int32    | Int64    | Int16     | Int8      | Notes                           |
| ------------------------- | -------- | -------- | -------- | -------- | --------- | --------- | ------------------------------- |
| **Binary Operations**     |          |          |          |          |           |           |
| add                       | ✅ SIMD   | ✅ SIMD   | ✅ SIMD   | ✅ SIMD   | ❌ Blocked | ❌ Blocked | 4x unrolled for f32/f64         |
| sub                       | ✅ SIMD   | ✅ SIMD   | ✅ SIMD   | ✅ SIMD   | ❌ Blocked | ❌ Blocked | 4x unrolled for f32/f64         |
| mul                       | ✅ SIMD   | ✅ SIMD   | ✅ SIMD   | ✅ SIMD   | ❌ Blocked | ❌ Blocked | 4x unrolled for f32/f64         |
| fdiv                      | ✅ SIMD   | ✅ SIMD   | N/A      | N/A      | N/A       | N/A       | Float only                      |
| idiv                      | N/A      | N/A      | ❌ Scalar | ❌ Scalar | ❌ Scalar  | ❌ Scalar  | No SIMD idiv in hardware        |
| mod                       | ❌ Scalar | ❌ Scalar | ❌ Scalar | ❌ Scalar | ❌ Scalar  | ❌ Scalar  | No SIMD mod in hardware         |
| pow                       | ❌ Scalar | ❌ Scalar | ❌ Scalar | ❌ Scalar | ❌ Scalar  | ❌ Scalar  | No SIMD pow in hardware         |
| min                       | ✅ SIMD   | ✅ SIMD   | ✅ SIMD   | ✅ SIMD   | ❌ Blocked | ❌ Blocked | 4x unrolled for f32/f64         |
| max                       | ✅ SIMD   | ✅ SIMD   | ✅ SIMD   | ✅ SIMD   | ❌ Blocked | ❌ Blocked | 4x unrolled for f32/f64         |
| **Unary Operations**      |          |          |          |          |           |           |
| neg                       | ✅ SIMD   | ✅ SIMD   | ✅ SIMD   | ✅ SIMD   | ❌ Blocked | ❌ Blocked | Int32/Int64 already use SIMD    |
| abs                       | ✅ SIMD   | ✅ SIMD   | ✅ SIMD   | ❌ Scalar | ❌ Blocked | ❌ Blocked | Int64 needs cross-platform impl |
| sqrt                      | ✅ SIMD   | ✅ SIMD   | N/A      | N/A      | N/A       | N/A       | Float only                      |
| recip                     | ✅ SIMD   | ✅ SIMD   | N/A      | N/A      | N/A       | N/A       | Float only                      |
| exp                       | ❌ Scalar | ❌ Scalar | N/A      | N/A      | N/A       | N/A       | No SIMD exp in hardware         |
| log                       | ❌ Scalar | ❌ Scalar | N/A      | N/A      | N/A       | N/A       | No SIMD log in hardware         |
| sin                       | ❌ Scalar | ❌ Scalar | N/A      | N/A      | N/A       | N/A       | No SIMD sin in hardware         |
| cos                       | ❌ Scalar | ❌ Scalar | N/A      | N/A      | N/A       | N/A       | No SIMD cos in hardware         |
| **Comparison Operations** |          |          |          |          |           |           |
| cmpeq                     | ❌ Scalar | ❌ Scalar | ❌ Scalar | ❌ Scalar | ❌ Scalar  | ❌ Scalar  | Output is bool, not mask        |
| cmpne                     | ❌ Scalar | ❌ Scalar | ❌ Scalar | ❌ Scalar | ❌ Scalar  | ❌ Scalar  | Output is bool, not mask        |
| cmplt                     | ❌ Scalar | ❌ Scalar | ❌ Scalar | ❌ Scalar | ❌ Scalar  | ❌ Scalar  | Output is bool, not mask        |
| cmple                     | ❌ Scalar | ❌ Scalar | ❌ Scalar | ❌ Scalar | ❌ Scalar  | ❌ Scalar  | Output is bool, not mask        |
| **Logical Operations**    |          |          |          |          |           |           |
| and                       | N/A      | N/A      | ✅ SIMD   | ✅ SIMD   | ❌ Blocked | ❌ Blocked | Int32/Int64 now use SIMD        |
| or                        | N/A      | N/A      | ✅ SIMD   | ✅ SIMD   | ❌ Blocked | ❌ Blocked | Int32/Int64 now use SIMD        |
| xor                       | N/A      | N/A      | ✅ SIMD   | ✅ SIMD   | ❌ Blocked | ❌ Blocked | Int32/Int64 now use SIMD        |
| **Reduce Operations**     |          |          |          |          |           |           |
| sum                       | ✅ SIMD   | ✅ SIMD   | ❌ Scalar | ❌ Scalar | ❌ Scalar  | ❌ Scalar  | 4x unrolled with hadd           |
| prod                      | ✅ SIMD   | ✅ SIMD   | ❌ Scalar | ❌ Scalar | ❌ Scalar  | ❌ Scalar  | 4x unrolled                     |
| max                       | ✅ SIMD   | ✅ SIMD   | ❌ Scalar | ❌ Scalar | ❌ Scalar  | ❌ Scalar  | 4x unrolled                     |
| min                       | ✅ SIMD   | ✅ SIMD   | ❌ Scalar | ❌ Scalar | ❌ Scalar  | ❌ Scalar  | 4x unrolled                     |

### Legend

- ✅ SIMD = Currently using SIMD optimization
- ❌ Scalar = Currently using scalar loops
- ❌ Blocked = SIMD ops exist but missing array load/store primitives
- N/A = Not applicable for this dtype

## Blocking Issue: Int8/Int16 Array Primitives

The SIMD types `int8x16#` and `int16x8#` exist in oxcaml with full arithmetic, comparison, and bitwise operations. However, **we cannot use them in kernels** because the array load/store primitives are missing from the compiler:

| Vector Type  | Array Load Primitive                   | Status        |
| ------------ | -------------------------------------- | ------------- |
| `int32x4#`   | `%caml_unboxed_int32_array_get128u#`   | ✅ Available   |
| `int64x2#`   | `%caml_unboxed_int64_array_get128u#`   | ✅ Available   |
| `float32x4#` | `%caml_unboxed_float32_array_get128u#` | ✅ Available   |
| `float64x2#` | `%caml_unboxed_float_array_get128u#`   | ✅ Available   |
| `int8x16#`   | `%caml_unboxed_int8_array_get128u#`    | ❌ **Missing** |
| `int16x8#`   | `%caml_unboxed_int16_array_get128u#`   | ❌ **Missing** |

**Action required**: These primitives need to be added to the oxcaml compiler before Int8/Int16 SIMD kernels can be implemented.

## Why Certain Kernels Don't Use SIMD

### 1. No Hardware Support (Cannot Be Fixed)

| Operation          | Reason                                                                                 |
| ------------------ | -------------------------------------------------------------------------------------- |
| idiv               | No SIMD integer division in SSE/AVX/NEON                                               |
| mod                | No SIMD modulo in hardware                                                             |
| pow                | No SIMD power function in hardware                                                     |
| exp, log, sin, cos | Transcendental functions - would need software polynomial approximations (SLEEF-style) |

### 2. Comparison Operations - Output Format Incompatibility

SIMD comparison operations (e.g., `Int32x4.cmpeq`) return a **mask** where each lane is either all 1s or all 0s. However, the nx-oxcaml comparison kernels output to a **bool array** (one byte per element with value 0 or 1).

Converting SIMD masks to bool arrays requires:
1. Performing the SIMD comparison to get a mask
2. Extracting each lane and converting to bool (0 or 1)
3. Writing to the output bool array

This conversion overhead may negate the SIMD benefit for comparisons.

### 3. Int64 abs - Missing Cross-Platform Implementation

- **NEON**: Has `neg` but no native `abs` for int64x2
- **SSE**: Has neither `neg` nor `abs` for int64x2

A portable implementation could use: `abs(x) = (x XOR mask) - mask` where `mask = cmpgt(zero, x)`, but this requires adding the function to the SIMD library first.

## Available Operations in oxcaml by Type

| Type      | Arithmetic              | Comparison         | Bitwise           | Min/Max | Shifts        | Special                        |
| --------- | ----------------------- | ------------------ | ----------------- | ------- | ------------- | ------------------------------ |
| Float32x4 | +, -, *, /              | eq, gt, ge, lt, le | via cast          | ✅       | N/A           | sqrt, rcp, rsqrt, round        |
| Float64x2 | +, -, *, /              | eq, gt, ge, lt, le | via cast          | ✅       | N/A           | sqrt, rcp, rsqrt, round        |
| Int32x4   | +, -, *, neg, abs       | eq, gt             | and, or, xor, not | ✅       | sll, srl, sra | hadd                           |
| Int64x2   | +, -                    | eq, gt             | and, or, xor, not | ✅       | sll, srl      | (no neg/abs in SSE)            |
| Int16x8   | +, -, mul_low, mul_high | eq, gt             | and, or, xor, not | ✅       | sll, srl, sra | abs, neg, hadd, saturating ops |
| Int8x16   | +, -, mul_hadd          | eq, gt             | and, or, xor, not | ✅       | N/A           | abs, neg, sad, saturating ops  |

## Recommendations

### Completed:

1. ✅ SIMD bitwise operations (and, or, xor) for Int32/Int64
2. ✅ SIMD reduction operations (sum, prod, max, min) for Float32/Float64 with 4x loop unrolling
3. ✅ 4x loop unrolling for binary ops (add, sub, mul, max, min) for Float32/Float64

### Remaining opportunities:

1. **Int64 abs SIMD** - Requires adding portable `abs` to Int64x2 module first
2. **Comparison ops** - Complex due to output format (mask vs bool array)
3. **Int32/Int64 reduction ops** - Could add SIMD for integer reductions

### Blocked by missing oxcaml primitives:

- Int8x16/Int16x8 SIMD for add, sub, min, max, and, or, xor kernels
- **Requires**: `%caml_unboxed_int8_array_get128u#` and `%caml_unboxed_int16_array_get128u#`

### Not possible without external work:

- Transcendentals (exp, log, sin, cos) - Would need SLEEF-style polynomial approximations
- Integer division - Must remain scalar
- Modulo - Must remain scalar
- Power - Could implement via exp/log once those are available

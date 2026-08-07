(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A 0-d operand broadcast against a full tensor reaches the C map kernels with
   an innermost step of 0 — the shape every [op_s] and every [op x (zeros_like
   x)] takes — and the kernels give it a dedicated branch that hoists the scalar
   out of the loop. That branch must produce EXACTLY what the other two branches
   produce: no reassociation, no widened intermediate, no reordered operands.

   Each case runs one computation in three shapes and compares the results bit
   for bit: contiguous against a materialized operand (the contiguous branch),
   contiguous against a 0-d broadcast (the splat branch), and a strided view of
   the same values against that broadcast (the generic byte walk). The operand
   values are the ones that would expose a difference — NaN, ±inf, ±0,
   subnormals, dtype extremes, and by-zero divisors — in both operand
   positions. *)

open Windtrap

(* Bit equality, not numeric equality: NaN must equal itself and -0. must differ
   from 0., which is the whole point of the comparison. Floats are compared
   through their f64 bits, which is injective on every stored float dtype since
   both sides widen identically. *)
let float_bits : float testable =
  Testable.make
    ~pp:(fun ppf v -> Format.fprintf ppf "%h" v)
    ~equal:(fun x y ->
      Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y))
    ()

let complex_bits : Complex.t testable =
  Testable.make
    ~pp:(fun ppf v -> Format.fprintf ppf "(%h, %h)" v.Complex.re v.Complex.im)
    ~equal:(fun x y ->
      Int64.equal
        (Int64.bits_of_float x.Complex.re)
        (Int64.bits_of_float y.Complex.re)
      && Int64.equal
           (Int64.bits_of_float x.Complex.im)
           (Int64.bits_of_float y.Complex.im))
    ()

let bit_testable (type a b) (dtype : (a, b) Nx.dtype) : a testable =
  match dtype with
  | Nx.Float16 -> float_bits
  | Nx.Float32 -> float_bits
  | Nx.Float64 -> float_bits
  | Nx.BFloat16 -> float_bits
  | Nx.Float8_e4m3 -> float_bits
  | Nx.Float8_e5m2 -> float_bits
  | Nx.Complex64 -> complex_bits
  | Nx.Complex128 -> complex_bits
  | Nx.Int4 -> int
  | Nx.UInt4 -> int
  | Nx.Int8 -> int
  | Nx.UInt8 -> int
  | Nx.Int16 -> int
  | Nx.UInt16 -> int
  | Nx.Int32 -> int32
  | Nx.UInt32 -> int32
  | Nx.Int64 -> int64
  | Nx.UInt64 -> int64
  | Nx.Bool -> bool

let same msg u v =
  equal ~msg (array (bit_testable (Nx.dtype u))) (Nx.to_array u) (Nx.to_array v)

(* The same values behind a step of 2 elements, so the innermost byte step is
   never the element size and the kernel falls to the generic walk. *)
let strided_like dtype values =
  let n = Array.length values in
  let wide = Array.make (2 * n) values.(0) in
  Array.iteri (fun i v -> wide.(2 * i) <- v) values;
  Nx.slice [ Nx.Rs (0, 2 * n, 2) ] (Nx.create dtype [| 2 * n |] wide)

let splat dtype n s = Nx.broadcast_to [| n |] (Nx.scalar dtype s)

let check_binop dtype values s (name, op) =
  let n = Array.length values in
  let x = Nx.create dtype [| n |] values in
  let xs = strided_like dtype values in
  let b = splat dtype n s in
  let full = Nx.full dtype [| n |] s in
  same (name ^ " rhs splat") (op x full) (op x b);
  same (name ^ " lhs splat") (op full x) (op b x);
  same (name ^ " rhs splat vs strided") (op x b) (op xs b);
  same (name ^ " lhs splat vs strided") (op b x) (op b xs)

let check_where dtype values s t =
  let n = Array.length values in
  let cond = Nx.create Nx.bool [| n |] (Array.init n (fun i -> i mod 3 <> 0)) in
  let x = Nx.create dtype [| n |] values in
  let xs = strided_like dtype values in
  let bs = splat dtype n s in
  let bt = splat dtype n t in
  let fs = Nx.full dtype [| n |] s in
  let ft = Nx.full dtype [| n |] t in
  same "where false-arm splat" (Nx.where cond x fs) (Nx.where cond x bs);
  same "where true-arm splat" (Nx.where cond fs x) (Nx.where cond bs x);
  same "where both splat" (Nx.where cond fs ft) (Nx.where cond bs bt);
  same "where false-arm splat vs strided" (Nx.where cond x bs)
    (Nx.where cond xs bs);
  same "where true-arm splat vs strided" (Nx.where cond bs x)
    (Nx.where cond bs xs)

(* ── Operand values ─────────────────────────────────────────────────────── *)

(* 67 elements: past any vector width, and prime so every kernel also runs a
   scalar tail. The head carries the values a wrong branch would expose; the
   rest is a ramp through both signs. *)
let n = 67

let float_values =
  let edges =
    [|
      Float.nan;
      Float.infinity;
      Float.neg_infinity;
      0.;
      -0.;
      1.;
      -1.;
      0.5;
      -0.5;
      Float.min_float;
      -.Float.min_float;
      Float.max_float;
      -.Float.max_float;
      4.9e-324 (* smallest subnormal *);
      65504. (* largest finite f16 *);
      6.1e-5;
    |]
  in
  Array.init n (fun i ->
      if i < Array.length edges then edges.(i)
      else float_of_int (i - 40) *. 0.375)

let complex_values =
  Array.init n (fun i ->
      { Complex.re = float_values.(i); im = float_values.((i + 7) mod n) })

let int_values width =
  let hi = (1 lsl (width - 1)) - 1 in
  let edges = [| 0; 1; -1; hi; -hi - 1; 2; -2; 7; -7 |] in
  Array.init n (fun i -> if i < Array.length edges then edges.(i) else i - 33)

let uint_values width =
  let hi = (1 lsl width) - 1 in
  let edges = [| 0; 1; hi; hi - 1; 2; 255 land hi |] in
  Array.init n (fun i ->
      if i < Array.length edges then edges.(i) else i land hi)

let i32_values =
  let edges = [| 0l; 1l; -1l; Int32.max_int; Int32.min_int; 2l; -2l |] in
  Array.init n (fun i ->
      if i < Array.length edges then edges.(i) else Int32.of_int (i - 33))

let i64_values =
  let edges = [| 0L; 1L; -1L; Int64.max_int; Int64.min_int; 2L; -2L |] in
  Array.init n (fun i ->
      if i < Array.length edges then edges.(i) else Int64.of_int (i - 33))

let bool_values = Array.init n (fun i -> i mod 7 < 3)

(* ── Per-category op lists ──────────────────────────────────────────────── *)

let arith =
  [
    ("add", Nx.add);
    ("sub", Nx.sub);
    ("mul", Nx.mul);
    ("div", Nx.div);
    ("pow", Nx.pow);
    ("maximum", Nx.maximum);
    ("minimum", Nx.minimum);
    ("mod", Nx.mod_);
  ]

let compares =
  [
    ("equal", Nx.equal);
    ("not_equal", Nx.not_equal);
    ("less", Nx.less);
    ("less_equal", Nx.less_equal);
    ("greater", Nx.greater);
    ("greater_equal", Nx.greater_equal);
  ]

let complex_arith =
  [ ("add", Nx.add); ("sub", Nx.sub); ("mul", Nx.mul); ("div", Nx.div) ]

let complex_compares = [ ("equal", Nx.equal); ("not_equal", Nx.not_equal) ]

let bitwise =
  [
    ("bitwise_and", Nx.bitwise_and);
    ("bitwise_or", Nx.bitwise_or);
    ("bitwise_xor", Nx.bitwise_xor);
  ]

(* ── Cases ──────────────────────────────────────────────────────────────── *)

(* Both a benign scalar and 0: by-zero is a defined, branchy result (idiv and
   mod return 0), and a branchy expression is exactly where a hoisted operand
   could compile differently. *)
let float_case name dtype =
  test name (fun () ->
      List.iter
        (fun s ->
          List.iter (check_binop dtype float_values s) arith;
          List.iter (check_binop dtype float_values s) compares;
          check_binop dtype float_values s ("atan2", Nx.atan2);
          check_where dtype float_values s (-.s))
        [ 2.5; 0.; -0.; Float.nan ])

let int_case name dtype values =
  test name (fun () ->
      List.iter
        (fun s ->
          List.iter (check_binop dtype values s) arith;
          List.iter (check_binop dtype values s) compares;
          List.iter (check_binop dtype values s) bitwise;
          check_where dtype values s (s + 1))
        [ 3; 0; 1 ])

let i32_case name dtype =
  test name (fun () ->
      List.iter
        (fun s ->
          List.iter (check_binop dtype i32_values s) arith;
          List.iter (check_binop dtype i32_values s) compares;
          List.iter (check_binop dtype i32_values s) bitwise;
          check_where dtype i32_values s (Int32.add s 1l))
        [ 3l; 0l; -1l ])

let i64_case name dtype =
  test name (fun () ->
      List.iter
        (fun s ->
          List.iter (check_binop dtype i64_values s) arith;
          List.iter (check_binop dtype i64_values s) compares;
          List.iter (check_binop dtype i64_values s) bitwise;
          check_where dtype i64_values s (Int64.add s 1L))
        [ 3L; 0L; -1L ])

let complex_case name dtype =
  test name (fun () ->
      List.iter
        (fun s ->
          List.iter (check_binop dtype complex_values s) complex_arith;
          List.iter (check_binop dtype complex_values s) complex_compares;
          check_where dtype complex_values s Complex.one)
        [ { Complex.re = 2.5; im = -1.25 }; Complex.zero ])

let float_cases =
  [
    float_case "float16" Nx.float16;
    float_case "float32" Nx.float32;
    float_case "float64" Nx.float64;
    float_case "bfloat16" Nx.bfloat16;
    float_case "float8_e4m3" Nx.float8_e4m3;
    float_case "float8_e5m2" Nx.float8_e5m2;
  ]

let int_cases =
  [
    int_case "int8" Nx.int8 (int_values 8);
    int_case "uint8" Nx.uint8 (uint_values 8);
    int_case "int16" Nx.int16 (int_values 16);
    int_case "uint16" Nx.uint16 (uint_values 16);
    i32_case "int32" Nx.int32;
    i32_case "uint32" Nx.uint32;
    i64_case "int64" Nx.int64;
    i64_case "uint64" Nx.uint64;
  ]

let complex_cases =
  [
    complex_case "complex64" Nx.complex64;
    complex_case "complex128" Nx.complex128;
  ]

let bool_cases =
  [
    test "bool" (fun () ->
        List.iter
          (fun s ->
            List.iter
              (check_binop Nx.bool bool_values s)
              [
                ("logical_and", Nx.logical_and);
                ("logical_or", Nx.logical_or);
                ("logical_xor", Nx.logical_xor);
                ("maximum", Nx.maximum);
                ("minimum", Nx.minimum);
              ];
            List.iter (check_binop Nx.bool bool_values s) compares;
            check_where Nx.bool bool_values s (not s))
          [ true; false ]);
  ]

(* A partially broadcast operand ([1;k] against [n;k]) keeps an innermost step
   of one element after coalescing, so it must stay on the contiguous branch and
   not be mistaken for a splat. Pinned here because the splat branches are
   selected on that step alone. *)
let row_broadcast_cases =
  [
    test "row broadcast is not a splat" (fun () ->
        let rows = 5 and cols = 13 in
        let row =
          Nx.create Nx.float32 [| 1; cols |]
            (Array.init cols (fun i -> float_values.(i)))
        in
        let x =
          Nx.create Nx.float32 [| rows; cols |]
            (Array.init (rows * cols) (fun i -> float_values.(i mod n)))
        in
        let tiled = Nx.broadcast_to [| rows; cols |] row in
        same "row rhs" (Nx.add x (Nx.contiguous tiled)) (Nx.add x row);
        same "row lhs" (Nx.add (Nx.contiguous tiled) x) (Nx.add row x));
  ]

let () =
  run "Nx Splat"
    [
      group "Float dtypes" float_cases;
      group "Integer dtypes" int_cases;
      group "Complex dtypes" complex_cases;
      group "Bool dtype" bool_cases;
      group "Partial broadcast" row_broadcast_cases;
    ]

(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The limits of each float dtype, against the values of its format's
   specification: IEEE 754 for float16, float32 and float64, bfloat16's
   truncated binary32, and OCP's float8 e4m3 and e5m2. *)

open Windtrap

type limits =
  | L : {
      dtype : (float, 'b) Nx_dtype.t;
      precision : int;
      epsilon : float;
      min_normal : float;
      max_finite : float;
    }
      -> limits

let limits =
  [
    L
      {
        dtype = Nx_dtype.float16;
        precision = 11;
        epsilon = 0x1p-10;
        min_normal = 0x1p-14;
        max_finite = 65504.;
      };
    L
      {
        dtype = Nx_dtype.float32;
        precision = 24;
        epsilon = 0x1p-23;
        min_normal = 0x1p-126;
        max_finite = 0x1.fffffep127;
      };
    L
      {
        dtype = Nx_dtype.float64;
        precision = 53;
        epsilon = 0x1p-52;
        min_normal = 0x1p-1022;
        max_finite = 0x1.fffffffffffffp1023;
      };
    L
      {
        dtype = Nx_dtype.bfloat16;
        precision = 8;
        epsilon = 0x1p-7;
        min_normal = 0x1p-126;
        max_finite = 0x1.fep127;
      };
    L
      {
        dtype = Nx_dtype.float8_e4m3;
        precision = 4;
        epsilon = 0x1p-3;
        min_normal = 0x1p-6;
        max_finite = 448.;
      };
    L
      {
        dtype = Nx_dtype.float8_e5m2;
        precision = 3;
        epsilon = 0x1p-2;
        min_normal = 0x1p-14;
        max_finite = 57344.;
      };
  ]

(* The value of [dt] whose bits follow those of the positive [x]. *)
let next_up (type b) (dt : (float, b) Nx_dtype.t) x =
  match dt with
  | Float64 -> Int64.float_of_bits (Int64.succ (Int64.bits_of_float x))
  | Float32 -> Int32.float_of_bits (Int32.succ (Int32.bits_of_float x))
  | dt ->
      let s = Nx_dtype.Scalar.of_dtype dt in
      Nx_dtype.Scalar.decode s (Nx_dtype.Scalar.encode s x + 1)

(* [x] stored in [dtype] and read back. *)
let stored dtype x = Nx.item [] (Nx.scalar dtype x)

let check (L l) =
  let name = Nx_dtype.to_string l.dtype in
  group name
    [
      test "the limits are the format's" (fun () ->
          equal ~msg:"precision" int l.precision (Nx_dtype.precision l.dtype);
          equal ~msg:"epsilon" float_exact l.epsilon (Nx_dtype.epsilon l.dtype);
          equal ~msg:"min_normal" float_exact l.min_normal
            (Nx_dtype.min_normal l.dtype);
          equal ~msg:"max_finite" float_exact l.max_finite
            (Nx_dtype.max_finite l.dtype));
      test "the integers up to 2^precision are held, and the next one is not"
        (fun () ->
          let top = Float.ldexp 1. (Nx_dtype.precision l.dtype) in
          equal float_exact (top -. 1.) (stored l.dtype (top -. 1.));
          equal float_exact top (stored l.dtype top);
          (* At float64, [top + 1] is not even an OCaml float. *)
          if Nx_dtype.precision l.dtype < 53 then
            not_equal float_exact (top +. 1.) (stored l.dtype (top +. 1.)));
      test "epsilon is the gap between one and the next value" (fun () ->
          let eps = Nx_dtype.epsilon l.dtype in
          equal float_exact (1. +. eps) (stored l.dtype (1. +. eps));
          equal float_exact 1. (stored l.dtype (1. +. (eps /. 2.))));
      test "the least normal value and the largest finite one are held"
        (fun () ->
          let lo = Nx_dtype.min_normal l.dtype in
          let hi = Nx_dtype.max_finite l.dtype in
          equal float_exact lo (stored l.dtype lo);
          equal float_exact hi (stored l.dtype hi);
          equal float_exact (-.hi) (stored l.dtype (-.hi)));
      test "the value whose bits follow the largest finite one is not finite"
        (fun () ->
          let hi = Nx_dtype.max_finite l.dtype in
          let after = next_up l.dtype hi in
          equal
            ~msg:(Printf.sprintf "after %h" hi)
            bool false (Float.is_finite after));
    ]

let () = exit (run "float limits" (List.map check limits))

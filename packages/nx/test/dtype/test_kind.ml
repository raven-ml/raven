(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Every dtype holds one of five kinds of number. [kind] names it, [is] tests
   it, and the [Float] arm of a match on [kind] reads the elements as
   [float]s. *)

open Windtrap

type dtype = D : ('a, 'b) Nx_dtype.t -> dtype
type kind = K : 'a Nx_dtype.kind -> kind

let name (type a) (k : a Nx_dtype.kind) =
  match k with
  | Float -> "float"
  | Complex -> "complex"
  | Signed -> "signed"
  | Unsigned -> "unsigned"
  | Boolean -> "boolean"

let kinds = [ K Nx_dtype.Float; K Complex; K Signed; K Unsigned; K Boolean ]

(* Each dtype with its kind as the interface lists it. *)
let dtypes =
  Nx_dtype.
    [
      (D float16, "float");
      (D float32, "float");
      (D float64, "float");
      (D bfloat16, "float");
      (D float8_e4m3, "float");
      (D float8_e5m2, "float");
      (D complex64, "complex");
      (D complex128, "complex");
      (D int4, "signed");
      (D int8, "signed");
      (D int16, "signed");
      (D int32, "signed");
      (D int64, "signed");
      (D uint4, "unsigned");
      (D uint8, "unsigned");
      (D uint16, "unsigned");
      (D uint32, "unsigned");
      (D uint64, "unsigned");
      (D bool, "boolean");
      (D bit, "boolean");
    ]

let dtype_name (D dt, _) = Nx_dtype.to_string dt

(* [x]'s first element as a [float] when its dtype is a float dtype. *)
let as_float (type a b) (x : (a, b) Nx.t) : float option =
  match Nx_dtype.kind (Nx.dtype x) with
  | Float -> Some (Nx.item [ 0 ] x)
  | _ -> None

let tests =
  [
    cases "kind names the dtype's kind of number" ~name:dtype_name dtypes
      (fun (D dt, expected) -> equal string expected (name (Nx_dtype.kind dt)));
    cases "is holds for the dtype's kind and no other" ~name:dtype_name dtypes
      (fun (D dt, expected) ->
        equal (list string) [ expected ]
          (List.filter_map
             (fun (K k) -> if Nx_dtype.is k dt then Some (name k) else None)
             kinds));
    cases "the Float arm reads a tensor's elements as floats" ~name:dtype_name
      dtypes (fun (D dt, expected) ->
        let x = Nx.ones dt [| 2 |] in
        let read = if expected = "float" then Some 1. else None in
        equal (option float_exact) read (as_float x));
  ]

let () = exit (run "kinds" [ group "Nx_dtype.kind" tests ])

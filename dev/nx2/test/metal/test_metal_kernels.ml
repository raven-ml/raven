(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* kernels.ml against kernels.h, as the C compiler reads it: every fact the host
   plans with is the header's. No GPU. *)

open Windtrap
module K = Kernels

(* The header's facts. Rows are the kernels' names; the others are triples: a
   constant's name and value, a field's name, offset and bytes, a struct's name
   and bytes. *)
module H = struct
  external kernels : unit -> string array = "nx_metal_test_kernel_rows"

  external constants : unit -> (string * int * int) array
    = "nx_metal_test_constants"

  external fields : unit -> (string * int * int) array = "nx_metal_test_fields"

  external structs : unit -> (string * int * int) array
    = "nx_metal_test_structs"
end

let named facts = List.map (fun (n, x, _) -> (n, x)) (Array.to_list facts)

(* Rows *)

(* An instance's name as kernels.h spells it: the family, the dtype and the
   orders, n as named and t transposed. *)

let dtype = function K.F32 -> "f32" | F16 -> "f16" | Bf16 -> "bf16"
let side t = if t then "t" else "n"
let order (o : K.order) = side o.a_t ^ side o.b_t

let name = function
  | K.Large (d, o) -> "contract_" ^ dtype d ^ "_" ^ order o
  | Small d -> "contract_" ^ dtype d ^ "_s"
  | Wide (d, o) -> "contract_" ^ dtype d ^ "_w" ^ order o
  | Checked d -> "contract_" ^ dtype d ^ "_l"
  | Int8 o -> "contract_i8_" ^ order o
  | Skinny (d, b_t) -> "skinny_" ^ dtype d ^ "_" ^ side b_t
  | Combine -> "contract_combine"
  | Int -> "contract_int"

(* Tests *)

let kernels () =
  equal (list string)
    (Array.to_list (H.kernels ()))
    (List.map fst (Array.to_list K.kernels))

let instances () =
  equal (list string)
    (List.map fst (Array.to_list K.kernels))
    (List.map (fun (_, i) -> name i) (Array.to_list K.kernels))

let constants () =
  equal
    (list (pair string int))
    (named (H.constants ()))
    [
      ("threads", K.threads);
      ("large", K.large);
      ("small", K.small);
      ("wide_m", K.wide_m);
      ("wide_n", K.wide_n);
      ("bk_half", K.bk_half);
      ("bk", K.bk);
      ("bk_wide", K.bk_wide);
      ("skinny_t", K.skinny_t);
      ("skinny_n", K.skinny_n);
      ("int_tile", K.int_tile);
      ("int_threads", K.int_threads);
      ("a_t", K.a_t);
      ("b_t", K.b_t);
      ("no_init", K.no_init);
    ]

(* The fields the stub lists are all of their struct's: in memory order, each
   starts where the one before ends, the first at 0, and the last ends at the
   struct's size. *)
let fields_tile () =
  let fields = Array.to_list (H.fields ()) in
  let check (s, size, _) =
    let mine (n, _, _) = String.starts_with ~prefix:(s ^ ".") n in
    let sorted =
      List.sort
        (fun (_, x, _) (_, y, _) -> Int.compare x y)
        (List.filter mine fields)
    in
    let starts, last =
      List.fold_left
        (fun (seen, at) (n, _, bytes) -> ((n, at) :: seen, at + bytes))
        ([], 0) sorted
    in
    equal ~msg:s
      (list (pair string int))
      (List.rev starts)
      (List.map (fun (n, x, _) -> (n, x)) sorted);
    equal ~msg:(s ^ "'s bytes") int size last
  in
  Array.iter check (H.structs ())

let offsets () =
  let module C = K.Contract_params in
  let module Q = K.Combine_params in
  equal
    (list (pair string int))
    (named (H.structs ()) @ named (H.fields ()))
    [
      ("nx_metal_contract", C.size);
      ("nx_metal_combine", Q.size);
      ("nx_metal_contract.a", C.a);
      ("nx_metal_contract.b", C.b);
      ("nx_metal_contract.init", C.init);
      ("nx_metal_contract.out", C.out);
      ("nx_metal_contract.a_batch", C.a_batch);
      ("nx_metal_contract.b_batch", C.b_batch);
      ("nx_metal_contract.init_batch", C.init_batch);
      ("nx_metal_contract.a_m", C.a_m);
      ("nx_metal_contract.a_k", C.a_k);
      ("nx_metal_contract.b_k", C.b_k);
      ("nx_metal_contract.b_n", C.b_n);
      ("nx_metal_contract.init_m", C.init_m);
      ("nx_metal_contract.init_n", C.init_n);
      ("nx_metal_contract.batch", C.batch);
      ("nx_metal_contract.m", C.m);
      ("nx_metal_contract.n", C.n);
      ("nx_metal_contract.k", C.k);
      ("nx_metal_contract.init_dtype", C.init_dtype);
      ("nx_metal_contract.out_dtype", C.out_dtype);
      ("nx_metal_contract.swizzle", C.swizzle);
      ("nx_metal_contract.dtype", C.dtype);
      ("nx_metal_contract.acc", C.acc);
      ("nx_metal_contract.order", C.order);
      ("nx_metal_combine.out", Q.out);
      ("nx_metal_combine.parts", Q.parts);
      ("nx_metal_combine.init", Q.init);
      ("nx_metal_combine.init_batch", Q.init_batch);
      ("nx_metal_combine.init_m", Q.init_m);
      ("nx_metal_combine.init_n", Q.init_n);
      ("nx_metal_combine.batch", Q.batch);
      ("nx_metal_combine.m", Q.m);
      ("nx_metal_combine.n", Q.n);
      ("nx_metal_combine.split", Q.split);
      ("nx_metal_combine.init_dtype", Q.init_dtype);
      ("nx_metal_combine.out_dtype", Q.out_dtype);
    ]

let () =
  exit
    (run "nx.metal kernels"
       [
         group "kernels.ml states kernels.h"
           [
             test "the kernels, row by row" kernels;
             test "each instance is the one its name states" instances;
             test "the geometry, the threads and the order bits" constants;
             test "the listed fields tile their structs" fields_tile;
             test "the structs' sizes and fields' offsets" offsets;
           ];
       ])

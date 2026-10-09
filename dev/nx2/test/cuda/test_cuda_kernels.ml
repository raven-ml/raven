(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* kernels.ml against kernels.h, as the C compiler reads it: every fact the host
   plans with is the header's. No GPU. *)

open Windtrap
module K = Kernels

(* The header's facts. Rows are the text of each X(...)'s arguments; the others
   are triples: a constant's name and value, a field's name, offset and bytes,
   a struct's name and bytes. *)
module H = struct
  external kernels : unit -> string array = "nx_cuda_test_kernel_rows"
  external tiles : unit -> string array = "nx_cuda_test_tile_rows"

  external constants : unit -> (string * int * int) array
    = "nx_cuda_test_constants"

  external fields : unit -> (string * int * int) array = "nx_cuda_test_fields"
  external structs : unit -> (string * int * int) array = "nx_cuda_test_structs"
end

let named facts = List.map (fun (n, x, _) -> (n, x)) (Array.to_list facts)

(* Rows *)

(* kernels.ml's values as kernels.h writes them. *)

let kind = function K.Bf16 -> "bf16" | F16 -> "f16" | S8 -> "s8" | Any -> "any"
let axis = function K.K -> "k" | M -> "m" | N -> "n"
let acc = function K.F32 -> "f32" | F64 -> "f64" | I64 -> "i64"

let tile = function
  | K.T128x128 -> "t128x128"
  | T128x256 -> "t128x256"
  | T64x64 -> "t64x64"
  | T16x64 -> "t16x64"

let kernel_row (name, instance) =
  let args =
    match instance with
    | K.Pack -> [ "PACK" ]
    | Mma (k, a, b, t) -> [ "MMA"; kind k; axis a; axis b; tile t ]
    | Simt (sum, side) -> [ "SIMT"; acc sum; string_of_int side ]
    | Skinny sum -> [ "SKINNY"; acc sum ]
  in
  String.concat ", " (name :: args)

let tile_row t =
  let s = K.shape t in
  Printf.sprintf "%s, %d, %d, %d, %d, %d, %d" (tile t) s.bm s.bn s.bkb s.wm s.wn
    s.stages

(* Tests *)

let kernels () =
  equal (list string)
    (Array.to_list (H.kernels ()))
    (List.map kernel_row (Array.to_list K.kernels))

let tiles () =
  equal (list string)
    (Array.to_list (H.tiles ()))
    (List.map tile_row (Array.to_list K.tiles))

let constants () =
  equal
    (list (pair string int))
    (named (H.constants ()))
    [
      ("a_vectors", K.a_vectors);
      ("b_vectors", K.b_vectors);
      ("b_across", K.b_across);
      ("y_whole", K.y_whole);
      ("skinny_rows", K.skinny_rows);
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
  let module P = K.Pack_params in
  equal
    (list (pair string int))
    (named (H.structs ()) @ named (H.fields ()))
    [
      ("contract_params", C.size);
      ("pack_params", P.size);
      ("contract_params.a", C.a);
      ("contract_params.b", C.b);
      ("contract_params.init", C.init);
      ("contract_params.y", C.y);
      ("contract_params.partials", C.partials);
      ("contract_params.tickets", C.tickets);
      ("contract_params.sa", C.sa);
      ("contract_params.sb", C.sb);
      ("contract_params.si", C.si);
      ("contract_params.sy", C.sy);
      ("contract_params.batch", C.batch);
      ("contract_params.m", C.m);
      ("contract_params.n", C.n);
      ("contract_params.k", C.k);
      ("contract_params.splits", C.splits);
      ("contract_params.a_dtype", C.a_dtype);
      ("contract_params.b_dtype", C.b_dtype);
      ("contract_params.init_dtype", C.init_dtype);
      ("contract_params.y_dtype", C.y_dtype);
      ("contract_params.acc_dtype", C.acc_dtype);
      ("contract_params.aligned", C.aligned);
      ("contract_params.unused", C.unused);
      ("pack_params.src", P.src);
      ("pack_params.dst", P.dst);
      ("pack_params.s", P.s);
      ("pack_params.lead", P.lead);
      ("pack_params.batch", P.batch);
      ("pack_params.rows", P.rows);
      ("pack_params.k", P.k);
      ("pack_params.dtype", P.dtype);
      ("pack_params.out", P.out);
      ("pack_params.bytes", P.bytes);
    ]

let () =
  exit
    (run "nx.cuda kernels"
       [
         group "kernels.ml states kernels.h"
           [
             test "the kernels, row by row" kernels;
             test "the tiles, row by row" tiles;
             test "the aligned bits and the skinny rows" constants;
             test "the listed fields tile their structs" fields_tile;
             test "the structs' sizes and fields' offsets" offsets;
           ];
       ])

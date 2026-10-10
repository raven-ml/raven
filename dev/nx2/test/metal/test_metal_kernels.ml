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

(* Stamp *)

(* A file of [contents]. *)
let file contents =
  let f = Filename.temp_file "stamp" "" in
  Out_channel.with_open_bin f (fun oc -> output_string oc contents);
  f

(* build.sh's digest of its sources: [md5 -q] of each file, a line each, then
   [md5 -q] of those lines. The values are that pipeline's. *)
let digest () =
  equal ~msg:"no file" string "d41d8cd98f00b204e9800998ecf8427e"
    (Stamp.digest []);
  equal ~msg:"one file" string "fd72b1ce6539aca765d3703d8397111f"
    (Stamp.digest [ file "a" ]);
  equal ~msg:"in order" string "ef0764cda821cae50c1d46f1511d3249"
    (Stamp.digest [ file "a"; file "" ])

(* A metallib's function list as its NAME tags spell it: a 16-bit
   little-endian length, then the name and a NUL. *)
let functions names =
  let tag f =
    let n = String.length f + 1 in
    "NAME" ^ String.make 1 (Char.chr (n land 0xff))
    ^ String.make 1 (Char.chr (n lsr 8))
    ^ f ^ "\000TYPE"
  in
  "MTLB" ^ String.concat "" (List.map tag names)

let stamp = Stamp.stamp "0123"

let names () =
  let has names f = Stamp.has (functions names) f in
  equal ~msg:"its stamp" bool true
    (has [ "contract_int"; stamp; "move" ] stamp);
  equal ~msg:"another stamp" bool false
    (has [ "contract_int"; Stamp.stamp "4567"; "move" ] stamp);
  equal ~msg:"no stamp" bool false (has [ "contract_int"; "move" ] stamp);
  equal ~msg:"a name only within another" bool false
    (has [ "contract_intx"; "xcontract_int"; stamp ] "contract_int")

let read file = In_channel.with_open_bin file In_channel.input_all

(* The metallibs, as the Metal compiler wrote them, name their kernels. *)
let harness () =
  let m = read "support/harness.metallib" in
  equal ~msg:"kernels it lacks" (list string) []
    (List.filter
       (fun f -> not (Stamp.has m f))
       [ "empty"; "move"; "probe_codec" ])

(* Off a Mac no Metal device opens, and a copied metallib serves as it is. *)
let library () =
  if not (Sys.file_exists "/System/Library/Frameworks/Metal.framework") then
    skip ~reason:"no Metal framework" ();
  let m = read "../../lib/metal/kernels/kernels.metallib" in
  equal ~msg:"kernels it lacks" (list string) []
    (List.filter
       (fun f -> not (Stamp.has m f))
       (Array.to_list (Array.map fst K.kernels)))

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
         group "metallibs"
           [
             test "the digest is build.sh's" digest;
             test "a metallib's functions read as whole NAME tags" names;
             test "the Metal compiler's names read as NAME tags" harness;
             test "nx.metal's metallib has every kernel kernels.ml names"
               library;
           ];
       ])

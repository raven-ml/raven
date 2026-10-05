(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The code objects the library carries: the archive kernels/tools/pack.ml
   writes, embedded by the assembler (nx_amd_stubs.c). Its index is read once; a
   member's bytes are copied out when a device first needs them. *)

external length : unit -> int = "caml_nx_amd_kernels_length"
external sub : int -> int -> string = "caml_nx_amd_kernels_sub"

let u64 s at = Int64.to_int (String.get_int64_le s at)

(* The members by key, as their offset and length. *)
let index =
  lazy
    (let n = u64 (sub 0 8) 0 in
     let t = Hashtbl.create n in
     let at = ref 8 in
     for _ = 1 to n do
       let k = u64 (sub !at 8) 0 in
       let entry = sub (!at + 8) (k + 16) in
       Hashtbl.replace t (String.sub entry 0 k) (u64 entry k, u64 entry (k + 8));
       at := !at + 24 + k
     done;
     if length () < !at then failwith "nx.amd: a truncated kernel archive";
     t)

(* The code object of [key], ["gfx12-generic/cast.float32.int8"], if the library
   carries it. *)
let find key =
  Option.map
    (fun (off, len) -> sub off len)
    (Hashtbl.find_opt (Lazy.force index) key)

(* The targets the library carries code objects for, and one code object of
   each. *)
let targets =
  lazy
    (Hashtbl.fold
       (fun key _ acc ->
         match String.index_opt key '/' with
         | Some i when not (List.mem_assoc (String.sub key 0 i) acc) ->
             (String.sub key 0 i, Option.get (find key)) :: acc
         | Some _ | None -> acc)
       (Lazy.force index) []
    |> List.sort compare)

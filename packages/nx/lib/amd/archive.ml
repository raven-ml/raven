(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The code objects the library carries: the archive kernels/tools/pack.ml
   writes, embedded by the assembler (nx_amd_stubs.c). Its index and targets are
   read when the module initialises, so that no domain forces a shared lazy; a
   member's bytes are copied out once, when a device first needs them, so that a
   device finds its load of them again as the same string. *)

external length : unit -> int = "caml_nx_amd_kernels_length"
external sub : int -> int -> string = "caml_nx_amd_kernels_sub"

let u64 s at = Int64.to_int (String.get_int64_le s at)

let malformed () = failwith "nx.amd: a malformed kernel archive"

(* Whether the [len] bytes from [off] lie in the archive. Offsets and lengths
   past [max_int] read as negative, so both are checked. *)
let inside off len = 0 <= off && 0 <= len && len <= length () - off

let slice off len = if inside off len then sub off len else malformed ()

(* The members by key, as their offset and length, each checked to lie in the
   archive so that [sub] copies only its bytes. *)
let index =
  let n = u64 (slice 0 8) 0 in
  (* Each entry of the index takes at least 24 bytes. *)
  if n < 0 || n > (length () - 8) / 24 then malformed ();
  let t = Hashtbl.create n in
  let at = ref 8 in
  for _ = 1 to n do
    let k = u64 (slice !at 8) 0 in
    if k < 0 then malformed ();
    let entry = slice (!at + 8) (k + 16) in
    let off = u64 entry k and len = u64 entry (k + 8) in
    if not (inside off len) then malformed ();
    Hashtbl.replace t (String.sub entry 0 k) (off, len);
    at := !at + 24 + k
  done;
  t

(* The code object of [key], ["gfx12-generic/cast.float32"], if the library
   carries it. *)
let find key =
  Option.map
    (fun (off, len) -> sub off len)
    (Hashtbl.find_opt index key)

(* A code object the library carries: its bytes and the names of its kernels. *)
type member = { binary : string; kernels : string list }

(* The code object of [key], if the library carries it, read once. *)
let member =
  let read = Hashtbl.create 64 and lock = Mutex.create () in
  fun key ->
    Mutex.protect lock @@ fun () ->
    match Hashtbl.find_opt read key with
    | Some m -> m
    | None ->
        let m =
          Option.map
            (fun binary ->
              match Rig_amd_abi.Code_object.of_string binary with
              | Ok co -> { binary; kernels = Rig_amd_abi.Code_object.kernels co }
              | Error e -> failwith (Printf.sprintf "nx.amd: %s: %s" key e))
            (find key)
        in
        Hashtbl.replace read key m;
        m

(* The targets the library carries code objects for, and one code object of
   each. *)
let targets =
  Hashtbl.fold
    (fun key _ acc ->
      match String.index_opt key '/' with
      | Some i when not (List.mem_assoc (String.sub key 0 i) acc) ->
          (String.sub key 0 i, Option.get (find key)) :: acc
      | Some _ | None -> acc)
    index []
  |> List.sort compare

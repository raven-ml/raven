(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module B = Rig.Buffer
module D = Nx_array.Dtype
module L = Nx_array.Layout

type bytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

(* Files *)

let open_file path =
  match Rig_disk.of_file path with
  | Ok f -> f
  | Error why -> raise (Sys_error why)

let string b ~pos n =
  if pos < 0 || n < 0 || pos > B.length b - n then
    Fail.failf "the file ends at byte %d, before byte %d" (B.length b)
      (if pos < 0 || n < 0 then 0 else pos + n);
  let s = Bytes.create n in
  B.blit_to_bytes b pos s 0 n;
  Bytes.unsafe_to_string s

(* The host's borrow of the file's pages, or, where the system cannot map the
   file, a copy of its bytes. *)
let pages file =
  match B.borrow Rig.host file with
  | Some h -> h
  | None ->
      let h = B.create Rig.host (B.length file) in
      B.copy ~src:file ~dst:h;
      h

let read b f = B.read b Bigarray.int8_unsigned f

(* Loading *)

let check_shape s =
  let bad why = Fail.failf "shape %s %s" (Fail.pp_shape s) why in
  if Array.length s > L.max_rank then bad "has more than 32 axes";
  let product n d =
    if d < 0 then bad "has a negative extent"
    else if d = 0 then n
    else if d > L.max_numel / n then bad "has more than 2^53 elements"
    else n * d
  in
  ignore (Array.fold_left product 1 s : int)

(* The alignment of [dt]'s elements in bytes: one component's width for a
   complex dtype. *)
let alignment (type v s) (dt : (v, s) D.t) =
  match D.kind dt with
  | D.Complex -> D.bits dt / 16
  | D.Float | D.Signed | D.Unsigned | D.Boolean -> Int.max 1 (D.bits dt / 8)

let copied b dt layout =
  let c = B.create Rig.host (B.length b) in
  B.copy ~src:b ~dst:c;
  Nx.Repr.unsafe_of_array Nx.Host.v (A.v dt layout c)

(* [value b dt layout] reads [b]'s own memory, through the host's borrow of a
   file's pages, where [b] starts on a multiple of [dt]'s alignment and the
   host maps it, and shares it; it reads a copy of [b]'s bytes otherwise,
   which it owns. *)
let value b dt layout =
  if B.offset b mod alignment dt = 0 then
    let a = A.v dt layout b in
    match A.borrow Rig.host a with
    | Some a -> Nx.Repr.of_array Nx.Host.v a
    | None -> Nx.Repr.unsafe_of_array Nx.Host.v (A.to_device Rig.host a)
  else copied b dt layout

(* Each element's bytes are a last axis of [w], split into components of [k]
   bytes, whose order is reversed: a complex number keeps its real part
   first. *)
let byteswap (type v s) (x : (v, s, Nx.host) Nx.t) : (v, s, Nx.host) Nx.t =
  let dt = Nx.dtype x in
  let k = alignment dt in
  if k = 1 then x
  else
    let s = Nx.shape x in
    let w = D.bits dt / 8 in
    let parts = Array.append s [| w / k; k |] in
    let bytes = Nx.reshape parts (Nx.bitcast Nx.uint8 x) in
    let swapped = Nx.flip ~axes:[ -1 ] bytes in
    Nx.bitcast dt (Nx.reshape (Array.append s [| w |]) swapped)

let little b dt layout =
  let x = value b dt layout in
  if Sys.big_endian then byteswap x else x

(* Saving *)

(* [as_bytes a] is the bytes of [a], C-contiguous and of a dtype of a byte or
   more, as a uint8 vector over [a]'s buffer. *)
let as_bytes a =
  let n = D.bytes (A.dtype a) (L.numel (A.layout a)) in
  let b = Option.get (A.bitcast D.Uint8 a) in
  Option.get (A.move (A.Move.Reshape (Iarray.of_array [| n |])) b)

(* [x]'s array is read in place where it is C-contiguous and in the host's byte
   order is the file's; otherwise a copy holds what the file holds. *)
let with_bytes ~by x f =
  let bytes a =
    A.read Bigarray.int8_unsigned (as_bytes a) (fun g ->
        f (Bigarray.array1_of_genarray g))
  in
  Fail.reading ~by (Nx.Repr.read x) @@ fun a ->
  let a = if L.is_contiguous (A.layout a) then a else A.copy a in
  if Sys.big_endian && alignment (A.dtype a) > 1 then
    let swapped = byteswap (Nx.Repr.unsafe_of_array Nx.Host.v (A.copy a)) in
    Nx.Repr.read swapped (fun b ->
        bytes (if L.is_contiguous (A.layout b) then b else A.copy b))
  else bytes a

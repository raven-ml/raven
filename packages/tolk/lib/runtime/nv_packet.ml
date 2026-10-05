(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

let u64 n = const ~dtype:Dtype.Uint64 (`Int (Bigint.of_int64_unsigned n))

let rec term = function
  | Nx_nv_packet.Value v -> v
  | Add (t, n) -> add (term t) (u64 n)
  | Shift (t, n) -> shr (term t) (int n)

let words =
  List.map (function
    | Nx_nv_packet.Dword n -> Hcq2.Queue.dword n
    | W32 t -> ccast (term t) Dtype.Uint32
    | W64 t -> ccast (term t) Dtype.Uint64)

let binary s = v Op.Binary ~arg:(Bytes s)

let region name blob patches =
  let patches = List.sort (fun (a, _) (b, _) -> Int.compare a b) patches in
  let bytes pos stop =
    if stop > pos then [ binary (String.sub blob pos (stop - pos)) ] else []
  in
  let rec words pos = function
    | [] -> bytes pos (String.length blob)
    | (off, w) :: rest ->
        bytes pos off @ (w :: words (off + Dtype.itemsize (dtype w)) rest)
  in
  v Op.Linear ~src:(words 0 patches) ~arg:(Region { name; align = 256 })

let unsigned = function
  | 1 -> Dtype.Uint8
  | 2 -> Dtype.Uint16
  | 4 -> Dtype.Uint32
  | _ -> Dtype.Uint64

(* The little-endian word of [n] bytes of [s] at [at]. *)
let word s at n =
  let w = ref 0L in
  for i = n - 1 downto 0 do
    w := Int64.(logor (shift_left !w 8) (of_int (Char.code s.[at + i])))
  done;
  !w

(* A hole's word: its term where its field fills the word, and else the term's
   field bits ored with the word's other bits. *)
let structure name (s : Ops.t Nx_nv_packet.structure) =
  let hole (h : Ops.t Nx_nv_packet.hole) =
    let n = List.find (fun n -> n * 8 >= h.bits) [ 1; 2; 4; 8 ] in
    let v = term h.value in
    let v =
      if n * 8 = h.bits then v
      else
        let mask = Int64.(sub (shift_left 1L h.bits) 1L) in
        let rest = Int64.logand (word s.bytes h.at n) (Int64.lognot mask) in
        bitwise_or (bitwise_and v (u64 mask)) (u64 rest)
    in
    (h.at, ccast v (unsigned n))
  in
  region name s.bytes (List.map hole s.holes)

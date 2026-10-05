(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

module Value = struct
  type t = Ops.t

  let add v n = add v (int ~dtype:Dtype.Uint64 n)
  let shift_right v n = shr v (int n)
end

let words =
  List.map (function
    | Nx_nv_packet.Dword n -> Hcq2.Queue.dword n
    | W32 v -> ccast v Dtype.Uint32
    | W64 v -> ccast v Dtype.Uint64)

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

let structure name (s : Ops.t Nx_nv_packet.structure) =
  region name s.bytes
    (List.map
       (fun (h : Ops.t Nx_nv_packet.hole) ->
         (h.at, ccast h.value (unsigned h.bytes)))
       s.holes)

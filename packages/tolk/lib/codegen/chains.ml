(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Ops

let chain = 256

(* A GPU thread's registers hold its accumulators, a nested sum two of each:
   past [held] bytes of them a thread's registers spill. On CUDA (255 a
   thread), a staged core's 128 float32 accumulators take 228 registers
   alone and spilled 560 bytes nested; a core's 80 took 240 nested, a
   blocked float32 tile's 64 took 207. Such a sum keeps its one running sum,
   and its bound is a single chain's. *)
let held = 320

let serial u =
  match axis_type u with
  | Axis_type.Weak | Loop | Upcast -> true
  | Global | Local | Warp | Placeholder -> false

(* Ranges nest by axis type, then identity. *)
let innermost rs =
  List.filter serial rs
  |> List.sort (fun a b ->
      let c = Axis_type.compare (axis_type b) (axis_type a) in
      if c <> 0 then c else Stdlib.compare (axis_id b) (axis_id a))

let cut ~lanes trips =
  let rec go p k = function
    | [] -> None
    | Some n :: rest when p * n <= chain -> go (p * n) (k + 1) rest
    | _ -> Some (k, chain / p)
  in
  go lanes 0 trips

let greatest_divisor n most =
  let rec go d = if n mod d = 0 then d else go (d - 1) in
  go (min n most)

let divisor n room =
  let d = greatest_divisor n room in
  if d > 1 && 2 * d >= room then d else room

let block ~lanes n =
  match cut ~lanes [ Some n ] with
  | None -> n
  | Some (_, room) -> if room < 2 then 1 else divisor n room

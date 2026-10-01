(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

type target = Upcast | Unroll | Local

type t =
  | Tc of { axis : int; tc_select : int; tc_opt : int; use_tc : int }
  | Split of { axis : int; amount : int; target : target; top : bool }
  | Padto of { axis : int; amount : int }
  | Swap of { axis : int; with_axis : int }

let axis = function
  | Tc { axis; _ } | Split { axis; _ } | Padto { axis; _ } | Swap { axis; _ } ->
      axis

let rank = function Tc _ -> 0 | Split _ -> 1 | Padto _ -> 2 | Swap _ -> 3
let target_rank = function Upcast -> 0 | Unroll -> 1 | Local -> 2

let args = function
  | Tc t -> [ t.tc_select; t.tc_opt; t.use_tc ]
  | Split s -> [ s.amount; target_rank s.target; Bool.to_int s.top ]
  | Padto p -> [ p.amount ]
  | Swap s -> [ s.with_axis ]

let compare o0 o1 =
  match Int.compare (rank o0) (rank o1) with
  | 0 -> (
      match Int.compare (axis o0) (axis o1) with
      | 0 -> List.compare Int.compare (args o0) (args o1)
      | c -> c)
  | c -> c

let equal o0 o1 = compare o0 o1 = 0

let target_name = function
  | Upcast -> "AxisType.UPCAST"
  | Unroll -> "AxisType.UNROLL"
  | Local -> "AxisType.LOCAL"

let tuple l = "(" ^ String.concat ", " l ^ ")"

let pp ppf o =
  let op, arg =
    match o with
    | Tc t ->
        ( "TC",
          tuple (List.map string_of_int [ t.tc_select; t.tc_opt; t.use_tc ]) )
    | Split s ->
        let parts = [ string_of_int s.amount; target_name s.target ] in
        ("SPLIT", tuple (if s.top then parts @ [ "True" ] else parts))
    | Padto p -> ("PADTO", string_of_int p.amount)
    | Swap s -> ("SWAP", string_of_int s.with_axis)
  in
  Format.fprintf ppf "Opt(op=OptOps.%s, axis=%d, arg=%s)" op (axis o) arg

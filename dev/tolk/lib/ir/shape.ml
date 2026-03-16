(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

type dim = Static of int | Symbol of { name : string; lo : int; hi : int }
type t = dim list

let scalar = []
let of_dims dims = List.map (fun dim -> Static dim) dims
let of_dim_list dims = dims
let dims t = t
let rank t = List.length t

let static_dims dims =
  let rec loop acc = function
    | [] -> Some (List.rev acc)
    | Static dim :: rest -> loop (dim :: acc) rest
    | Symbol _ :: _ -> None
  in
  loop [] dims

let pp_dim fmt = function
  | Static dim -> Format.pp_print_int fmt dim
  | Symbol { name; lo; hi } -> Format.fprintf fmt "%s[%d..%d]" name lo hi

let pp fmt dims =
  Format.fprintf fmt "[%a]"
    (Format.pp_print_list
       ~pp_sep:(fun fmt () -> Format.fprintf fmt ", ")
       pp_dim)
    dims

(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

type dim = Static of int | Symbol of { name : string; lo : int; hi : int }
type t = dim list

let scalar = []
let of_dims ns = List.map (fun n -> Static n) ns
let of_dim_list ds = ds
let dims s = s
let rank s = List.length s

let static_dims s =
  let rec loop acc = function
    | [] -> Some (List.rev acc)
    | Static n :: rest -> loop (n :: acc) rest
    | Symbol _ :: _ -> None
  in
  loop [] s

let pp_dim ppf = function
  | Static n -> Format.pp_print_int ppf n
  | Symbol { name; lo; hi } -> Format.fprintf ppf "%s[%d..%d]" name lo hi

let pp_sep ppf () = Format.pp_print_string ppf ", "
let pp ppf s = Format.fprintf ppf "[%a]" (Format.pp_print_list ~pp_sep pp_dim) s

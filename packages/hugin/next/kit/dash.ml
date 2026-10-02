(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = float list

let v lengths =
  List.iter
    (fun l ->
      if not (Float.is_finite l && l >= 0.) then
        invalid_arg
          (Printf.sprintf "Dash.v: the length %g is not finite and not negative"
             l))
    lengths;
  match lengths with
  | [] -> lengths
  | _ :: _ ->
      if List.fold_left ( +. ) 0. lengths = 0. then
        invalid_arg "Dash.v: the lengths sum to 0";
      lengths

let solid = []
let dashed = [ 4.; 2. ]
let dotted = [ 1.; 2. ]
let dash_dot = [ 4.; 2.; 1.; 2. ]
let all = [ solid; dashed; dotted; dash_dot ]
let lengths d = d
let equal = List.equal Float.equal

let pp ppf = function
  | [] -> Format.pp_print_string ppf "solid"
  | d ->
      Format.fprintf ppf "@[%a@]"
        (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf l ->
             Format.fprintf ppf "%g" l))
        d

(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | Group_id of int
  | Local_id of int
  | Global_idx of int

let axis = function
  | Group_id axis | Local_id axis | Global_idx axis -> axis

let to_special_name = function
  | Group_id axis -> Printf.sprintf "gidx%d" axis
  | Local_id axis -> Printf.sprintf "lidx%d" axis
  | Global_idx axis -> Printf.sprintf "idx%d" axis

let parse_prefix name prefix make =
  let n = String.length prefix in
  if String.length name >= n && String.sub name 0 n = prefix then
    match int_of_string_opt (String.sub name n (String.length name - n)) with
    | Some axis -> Some (make axis)
    | None -> None
  else None

let of_special_name name =
  match parse_prefix name "gidx" (fun axis -> Group_id axis) with
  | Some _ as dim -> dim
  | None ->
      (match parse_prefix name "lidx" (fun axis -> Local_id axis) with
       | Some _ as dim -> dim
       | None -> parse_prefix name "idx" (fun axis -> Global_idx axis))

let rank = function
  | Group_id _ -> 0
  | Local_id _ -> 1
  | Global_idx _ -> 2

let compare a b = Stdlib.compare (rank a, axis a) (rank b, axis b)
let equal a b = compare a b = 0

let compare_special_name a b =
  match of_special_name a, of_special_name b with
  | Some a, Some b -> compare a b
  | Some _, None -> -1
  | None, Some _ -> 1
  | None, None -> String.compare a b
